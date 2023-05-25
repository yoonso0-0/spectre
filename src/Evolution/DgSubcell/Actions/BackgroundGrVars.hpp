// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/OnSubcellFaces.hpp"
#include "Evolution/Initialization/InitialData.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Time/Tags.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace evolution::dg::subcell::Actions {

template <typename System, bool used_in_evolution_phase>
struct BackgroundGrVars {
  static constexpr size_t Dim = System::volume_dim;

  using simple_tags_from_options = tmpl::list<::Tags::Time>;

  // FIXME : should I include all-the-gr-tags like DG version of this action?
  using gr_tag = typename System::spacetime_variables_tag;
  using subcell_gr_tag = evolution::dg::subcell::Tags::Inactive<gr_tag>;
  using subcell_faces_gr_tag = evolution::dg::subcell::Tags::OnSubcellFaces<
      typename System::flux_spacetime_variables_tag, Dim>;

  using GrVars = typename gr_tag::type;
  using SubcellFaceGrVars = typename subcell_faces_gr_tag::type;

  using simple_tags = tmpl::list<subcell_gr_tag, subcell_faces_gr_tag>;
  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    if constexpr (db::tag_is_retrievable_v<
                      evolution::initial_data::Tags::InitialData,
                      db::DataBox<DbTagsList>>) {
      using derived_classes =
          tmpl::at<typename Metavariables::factory_creation::factory_classes,
                   evolution::initial_data::InitialData>;
      call_with_dynamic_type<void, derived_classes>(
          &db::get<evolution::initial_data::Tags::InitialData>(box),
          [&box](const auto* const data_or_solution) {
            impl(make_not_null(&box), *data_or_solution);
          });
    } else if constexpr (db::tag_is_retrievable_v<
                             ::Tags::AnalyticSolutionOrData,
                             db::DataBox<DbTagsList>>) {
      impl(make_not_null(&box), db::get<::Tags::AnalyticSolutionOrData>(box));
    } else {
      ERROR(
          "Either ::Tags::AnalyticSolutionOrData or "
          "evolution::initial_data::Tags::InitialData must be in the "
          "DataBox.");
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }

 private:
  template <typename DbTagsList, typename T>
  static void impl(const gsl::not_null<db::DataBox<DbTagsList>*> box,
                   const T& solution_or_data) {
    const double time = db::get<::Tags::Time>(*box);
    const auto& subcell_mesh =
        get<evolution::dg::subcell::Tags::Mesh<Dim>>(*box);
    const auto& subcell_inertial_coords = db::get<
        evolution::dg::subcell::Tags::Coordinates<Dim, Frame::Inertial>>(*box);

    // Evaluate (or initialize) cell-centered values
    if constexpr (used_in_evolution_phase) {
      const auto& element_id = get<domain::Tags::Element<Dim>>(*box).id();
      const size_t block_id = element_id.block_id();
      const Block<Dim>& block =
          get<domain::Tags::Domain<Dim>>(*box).blocks()[block_id];

      if (block.is_time_dependent()) {
        // Q. Why should we use gr_tag here for the correct behavior?
        db::mutate<gr_tag>(box, [&time, &subcell_inertial_coords,
                                 &solution_or_data](auto subcell_gr_vars) {
          subcell_gr_vars->assign_subset(
              evolution::Initialization::initial_data(
                  solution_or_data, subcell_inertial_coords, time,
                  typename GrVars::tags_list{}));
        });
      }
    } else {
      // We need an allocation of GR variables only in the initialization phase.
      GrVars subcell_gr_vars{subcell_mesh.number_of_grid_points()};

      subcell_gr_vars.assign_subset(evolution::Initialization::initial_data(
          solution_or_data, subcell_inertial_coords, time,
          typename GrVars::tags_list{}));

      ::Initialization::mutate_assign<tmpl::list<subcell_gr_tag>>(
          box, std::move(subcell_gr_vars));
    }

    // Now evaluate (or initialize) face-centered values
    ASSERT(Mesh<Dim>(subcell_mesh.extents(0), subcell_mesh.basis(0),
                     subcell_mesh.quadrature(0)) == subcell_mesh,
           "The subcell mesh must have isotropic basis, quadrature. and "
           "extents but got "
               << subcell_mesh);
    const auto& logical_to_grid_map =
        db::get<domain::Tags::ElementMap<Dim, Frame::Grid>>(*box);
    const auto& grid_to_inertial_map =
        db::get<domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                            Frame::Inertial>>(
            *box);
    const auto& functions_of_time =
        db::get<domain::Tags::FunctionsOfTime>(*box);

    SubcellFaceGrVars subcell_face_gr_vars{};

    for (size_t dim = 0; dim < Dim; ++dim) {
      const auto basis = make_array<Dim>(subcell_mesh.basis(0));
      auto quadrature = make_array<Dim>(subcell_mesh.quadrature(0));
      auto extents = make_array<Dim>(subcell_mesh.extents(0));
      gsl::at(extents, dim) = subcell_mesh.extents(0) + 1;
      gsl::at(quadrature, dim) = Spectral::Quadrature::FaceCentered;
      const Mesh<Dim> face_centered_mesh{extents, basis, quadrature};
      const auto face_centered_logical_coords =
          logical_coordinates(face_centered_mesh);
      const auto face_centered_inertial_coords = grid_to_inertial_map(
          logical_to_grid_map(face_centered_logical_coords), time,
          functions_of_time);

      if constexpr (used_in_evolution_phase) {
        db::mutate<subcell_faces_gr_tag>(
            box,
            [&dim, &time, &face_centered_inertial_coords, &solution_or_data](
                const gsl::not_null<SubcellFaceGrVars*> subcell_gr_vars) {
              gsl::at(*subcell_gr_vars, dim)
                  .assign_subset(evolution::Initialization::initial_data(
                      solution_or_data, face_centered_inertial_coords, time,
                      typename SubcellFaceGrVars::value_type::tags_list{}));
            });
      } else {
        // We need an allocation of GR variables only in the initialization
        // phase.
        gsl::at(subcell_face_gr_vars, dim)
            .initialize(face_centered_mesh.number_of_grid_points());
        gsl::at(subcell_face_gr_vars, dim)
            .assign_subset(evolution::Initialization::initial_data(
                solution_or_data, face_centered_inertial_coords, time,
                typename SubcellFaceGrVars::value_type::tags_list{}));
      }
    }

    if constexpr (not used_in_evolution_phase) {
      ::Initialization::mutate_assign<tmpl::list<subcell_faces_gr_tag>>(
          box, std::move(subcell_face_gr_vars));
    }
  }
};

}  // namespace evolution::dg::subcell::Actions
