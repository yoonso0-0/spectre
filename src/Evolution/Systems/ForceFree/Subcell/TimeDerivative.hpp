// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/TaggedContainers.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DgSubcell/CartesianFluxDivergence.hpp"
#include "Evolution/DgSubcell/ComputeBoundaryTerms.hpp"
#include "Evolution/DgSubcell/CorrectPackagedData.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/ReconstructionOrder.hpp"
#include "Evolution/DgSubcell/Tags/CellCenteredFlux.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/Jacobians.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/OnSubcellFaces.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/PackageDataImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/Systems/ForceFree/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/ForceFree/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/BoundaryConditionGhostData.hpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/ForceFree/Fluxes.hpp"
#include "Evolution/Systems/ForceFree/Sources.hpp"
#include "Evolution/Systems/ForceFree/Subcell/ComputeFluxes.hpp"
#include "Evolution/Systems/ForceFree/System.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "NumericalAlgorithms/FiniteDifference/DerivativeOrder.hpp"
#include "NumericalAlgorithms/FiniteDifference/HighOrderFluxCorrection.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

namespace ForceFree::subcell {
/*!
 * \brief Compute the time derivative on the subcell grid using FD
 * reconstruction.
 *
 * The code makes the following unchecked assumptions:
 * - Assumes Cartesian coordinates with a diagonal Jacobian matrix
 * from the logical to the inertial frame
 *
 */
struct TimeDerivative {
  template <typename DbTagsList>
  static void apply(const gsl::not_null<db::DataBox<DbTagsList>*> box) {
    using evolved_vars_tags = typename System::variables_tag::tags_list;
    using fluxes_tags = typename Fluxes::return_tags;

    const Mesh<3>& dg_mesh = db::get<domain::Tags::Mesh<3>>(*box);
    const Mesh<3> subcell_mesh =
        db::get<evolution::dg::subcell::Tags::Mesh<3>>(*box);
    ASSERT(
        subcell_mesh == Mesh<3>(subcell_mesh.extents(0), subcell_mesh.basis(0),
                                subcell_mesh.quadrature(0)),
        "The subcell/FD mesh must be isotropic for the FD time derivative but "
        "got "
            << subcell_mesh);

    const size_t num_reconstructed_pts =
        (subcell_mesh.extents(0) + 1) *
        subcell_mesh.extents().slice_away(0).product();

    const ForceFree::fd::Reconstructor& recons =
        db::get<ForceFree::fd::Tags::Reconstructor>(*box);

    const auto& boundary_correction =
        db::get<evolution::Tags::BoundaryCorrection<System>>(*box);
    using derived_boundary_corrections =
        typename std::decay_t<decltype(boundary_correction)>::creatable_classes;

    // Project DG mesh velocity onto subcell if needed
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
        dg_volume_mesh_velocity = db::get<domain::Tags::MeshVelocity<3>>(*box);
    const std::optional<Scalar<DataVector>>& div_dg_mesh_velocity =
        db::get<domain::Tags::DivMeshVelocity>(*box);

    // boundary correction terms of evolved variables on subcell interfaces
    std::array<Variables<evolved_vars_tags>, 3> fd_boundary_corrections{};

    // Check if element is at external boundary or FD is enabled at there.
    const Element<3>& element = db::get<domain::Tags::Element<3>>(*box);
    const bool element_is_interior = element.external_boundaries().empty();
    constexpr bool subcell_enabled_at_external_boundary =
        std::decay_t<decltype(db::get<Parallel::Tags::Metavariables>(
            *box))>::SubcellOptions::subcell_enabled_at_external_boundary;
    ASSERT(element_is_interior or subcell_enabled_at_external_boundary,
           "Subcell time derivative is called at a boundary element while "
           "using subcell is disabled at external boundaries."
           "ElementID "
               << element.id());
    // If the element has external boundaries and subcell is enabled for
    // boundary elements, compute FD ghost data with a given boundary condition.
    if constexpr (subcell_enabled_at_external_boundary) {
      if (not element.external_boundaries().empty()) {
        fd::BoundaryConditionGhostData::apply(box, element, recons);
      }
    }

    // Higher order FD corrections and recons order data
    const auto fd_derivative_order =
        db::get<evolution::dg::subcell::Tags::SubcellOptions<3>>(*box)
            .finite_difference_derivative_order();

    std::optional<std::array<std::vector<std::uint8_t>, 3>>
        reconstruction_order_data{};
    std::optional<std::array<gsl::span<std::uint8_t>, 3>>
        reconstruction_order{};
    if (static_cast<int>(fd_derivative_order) < 0) {
      reconstruction_order_data = make_array<3>(std::vector<std::uint8_t>(
          (subcell_mesh.extents(0) + 2) * subcell_mesh.extents(1) *
              subcell_mesh.extents(2),
          std::numeric_limits<std::uint8_t>::max()));
      reconstruction_order = std::array<gsl::span<std::uint8_t>, 3>{};
      for (size_t i = 0; i < 3; ++i) {
        gsl::at(reconstruction_order.value(), i) = gsl::make_span(
            gsl::at(reconstruction_order_data.value(), i).data(),
            gsl::at(reconstruction_order_data.value(), i).size());
      }
    }

    call_with_dynamic_type<void, derived_boundary_corrections>(
        &boundary_correction, [&](const auto* derived_correction) {
          using DerivedCorrection = std::decay_t<decltype(*derived_correction)>;

          using dg_package_data_temporary_tags =
              typename DerivedCorrection::dg_package_data_temporary_tags;

          using dg_package_data_argument_tags = tmpl::append<
              evolved_vars_tags, fluxes_tags,
              tmpl::remove_duplicates<tmpl::push_back<
                  dg_package_data_temporary_tags,
                  gr::Tags::SpatialMetric<DataVector, 3>,
                  gr::Tags::SqrtDetSpatialMetric<DataVector>,
                  gr::Tags::InverseSpatialMetric<DataVector, 3,
                                                 Frame::Inertial>,
                  evolution::dg::Actions::detail::NormalVector<3>>>>;

          // Data that needs to be reconstructed for calling dg_package_data().
          //
          // First, we do allocation of Variables object with extra buffer for
          // storing TildeJ; it is not an argument of dg_package_data()
          // function, but needs to be reconstructed for computing fluxes on
          // faces.
          auto package_data_argvars_lower_face_with_tildej = make_array<3>(
              Variables<tmpl::append<tmpl::list<ForceFree::Tags::TildeJ>,
                                     dg_package_data_argument_tags>>(
                  num_reconstructed_pts));
          auto package_data_argvars_upper_face_with_tildej = make_array<3>(
              Variables<tmpl::append<tmpl::list<ForceFree::Tags::TildeJ>,
                                     dg_package_data_argument_tags>>(
                  num_reconstructed_pts));
          // then create a `view` of the allocation excluding the TildeJ tag,
          // since boundary correction takes the flux F^i(TildeQ) as argument
          // but not TildeJ.
          auto package_data_argvars_lower_face =
              std::array<Variables<dg_package_data_argument_tags>, 3>{};
          auto package_data_argvars_upper_face =
              std::array<Variables<dg_package_data_argument_tags>, 3>{};
          for (size_t d = 0; d < 3; ++d) {
            gsl::at(package_data_argvars_lower_face, d) =
                gsl::at(package_data_argvars_lower_face_with_tildej, d)
                    .template reference_subset<dg_package_data_argument_tags>();
            gsl::at(package_data_argvars_upper_face, d) =
                gsl::at(package_data_argvars_upper_face_with_tildej, d)
                    .template reference_subset<dg_package_data_argument_tags>();
          }

          // Copy over the face values of the metric quantities which are
          // required for computing fluxes
          using face_spacetime_vars_to_copy =
              System::flux_spacetime_variables_tag::tags_list;
          tmpl::for_each<face_spacetime_vars_to_copy>(
              [&package_data_argvars_lower_face,
               &package_data_argvars_upper_face,
               &spacetime_vars_on_face =
                   db::get<evolution::dg::subcell::Tags::OnSubcellFaces<
                       typename System::flux_spacetime_variables_tag, 3>>(
                       *box)](auto tag_v) {
                using tag = tmpl::type_from<decltype(tag_v)>;
                for (size_t d = 0; d < 3; ++d) {
                  get<tag>(gsl::at(package_data_argvars_lower_face, d)) =
                      get<tag>(gsl::at(spacetime_vars_on_face, d));
                  get<tag>(gsl::at(package_data_argvars_upper_face, d)) =
                      get<tag>(gsl::at(spacetime_vars_on_face, d));
                }
              });

          // Perform FD reconstruction of variables on cell interfaces. Note
          // that we are using the argvars object with TildeJ buffer so that
          // TildeJ can be reconstructed in this step.
          auto recons_vars_lower_face =
              std::array<Variables<ForceFree::fd::tags_list_for_reconstruction>,
                         3>{};
          auto recons_vars_upper_face =
              std::array<Variables<ForceFree::fd::tags_list_for_reconstruction>,
                         3>{};
          for (size_t d = 0; d < 3; ++d) {
            gsl::at(recons_vars_lower_face, d) =
                gsl::at(package_data_argvars_lower_face_with_tildej, d)
                    .template reference_subset<
                        ForceFree::fd::tags_list_for_reconstruction>();
            gsl::at(recons_vars_upper_face, d) =
                gsl::at(package_data_argvars_upper_face_with_tildej, d)
                    .template reference_subset<
                        ForceFree::fd::tags_list_for_reconstruction>();
          }
          call_with_dynamic_type<
              void, typename ForceFree::fd::Reconstructor::creatable_classes>(
              &recons, [&box, &recons_vars_lower_face,
                        &recons_vars_upper_face](const auto& reconstructor) {
                db::apply<typename std::decay_t<
                    decltype(*reconstructor)>::reconstruction_argument_tags>(
                    [&recons_vars_lower_face, &recons_vars_upper_face,
                     &reconstructor](const auto&... args) {
                      reconstructor->reconstruct(
                          make_not_null(&recons_vars_lower_face),
                          make_not_null(&recons_vars_upper_face), args...);
                    },
                    *box);
              });

          // Variables to store packaged data. Allocate outside of loop to
          // reduce allocations.
          using dg_package_field_tags =
              typename DerivedCorrection::dg_package_field_tags;
          Variables<dg_package_field_tags> upper_packaged_data{
              num_reconstructed_pts};
          Variables<dg_package_field_tags> lower_packaged_data{
              num_reconstructed_pts};

          for (size_t i = 0; i < 3; ++i) {
            compute_fluxes(make_not_null(
                &gsl::at(package_data_argvars_upper_face_with_tildej, i)));
            compute_fluxes(make_not_null(
                &gsl::at(package_data_argvars_lower_face_with_tildej, i)));

            // Now that we have computed fluxes, from here we can proceed
            // without caring about the TildeJ part of the allocation.
            auto& vars_upper_face = gsl::at(package_data_argvars_upper_face, i);
            auto& vars_lower_face = gsl::at(package_data_argvars_lower_face, i);

            // Create face-centered subcell mesh extents toward the i-th
            // direction
            Index<3> subcell_face_centered_mesh_extents =
                subcell_mesh.extents();
            ++subcell_face_centered_mesh_extents[i];

            // Apply mesh velocity corrections to fluxes if needed
            std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>
                subcell_mesh_velocity_on_faces = {};
            if (dg_volume_mesh_velocity.has_value()) {
              // Project mesh velocity on face mesh
              // Can we get away with only doing the normal component? It
              // is also used in the packaged data...
              subcell_mesh_velocity_on_faces =
                  tnsr::I<DataVector, 3, Frame::Inertial>{
                      num_reconstructed_pts};
              for (size_t j = 0; j < 3; j++) {
                // j^th component of the velocity on the i^th directed face
                subcell_mesh_velocity_on_faces.value().get(j) =
                    evolution::dg::subcell::fd::project_to_faces(
                        dg_volume_mesh_velocity.value().get(j), dg_mesh,
                        subcell_face_centered_mesh_extents, i);
              }

              tmpl::for_each<evolved_vars_tags>(
                  [&vars_upper_face, &vars_lower_face,
                   &subcell_mesh_velocity_on_faces](auto tag_v) {
                    using tag = tmpl::type_from<decltype(tag_v)>;
                    using flux_tag =
                        ::Tags::Flux<tag, tmpl::size_t<3>, Frame::Inertial>;
                    using FluxTensor = typename flux_tag::type;
                    const auto& var_upper = get<tag>(vars_upper_face);
                    const auto& var_lower = get<tag>(vars_lower_face);
                    auto& flux_upper = get<flux_tag>(vars_upper_face);
                    auto& flux_lower = get<flux_tag>(vars_lower_face);
                    for (size_t storage_index = 0;
                         storage_index < var_upper.size(); ++storage_index) {
                      const auto tensor_index =
                          var_upper.get_tensor_index(storage_index);
                      for (size_t j = 0; j < 3; j++) {
                        const auto flux_storage_index =
                            FluxTensor::get_storage_index(
                                prepend(tensor_index, j));
                        flux_upper[flux_storage_index] -=
                            subcell_mesh_velocity_on_faces.value().get(j) *
                            var_upper[storage_index];
                        flux_lower[flux_storage_index] -=
                            subcell_mesh_velocity_on_faces.value().get(j) *
                            var_lower[storage_index];
                      }
                    }
                  });
            }

            // Normal vectors in curved spacetime normalized by inverse
            // spatial metric. Since we assume a Cartesian grid, this is
            // relatively easy. Note that we use the sign convention on
            // the normal vectors to be compatible with DG.
            //
            // Note that these normal vectors are on all faces inside the DG
            // element since there are a bunch of subcells. We don't use the
            // NormalCovectorAndMagnitude tag in the DataBox right now to avoid
            // conflicts with the DG solver. We can explore in the future if
            // it's possible to reuse that allocation.
            const Scalar<DataVector> normalization{
                sqrt(get<gr::Tags::InverseSpatialMetric<
                         DataVector, 3, Frame::Inertial>>(vars_upper_face)
                         .get(i, i))};

            tnsr::i<DataVector, 3, Frame::Inertial> lower_outward_conormal{
                num_reconstructed_pts, 0.0};
            lower_outward_conormal.get(i) = 1.0 / get(normalization);

            tnsr::i<DataVector, 3, Frame::Inertial> upper_outward_conormal{
                num_reconstructed_pts, 0.0};
            upper_outward_conormal.get(i) = -lower_outward_conormal.get(i);
            // Note: we probably should compute the normal vector in addition to
            // the co-vector. Not a huge issue since we'll get an FPE right now
            // if it's used by a Riemann solver.

            // Compute the packaged data
            using dg_package_data_projected_tags = tmpl::append<
                evolved_vars_tags, fluxes_tags, dg_package_data_temporary_tags,
                typename DerivedCorrection::dg_package_data_primitive_tags>;

            evolution::dg::Actions::detail::dg_package_data<System>(
                make_not_null(&upper_packaged_data),
                dynamic_cast<const DerivedCorrection&>(boundary_correction),
                vars_upper_face, upper_outward_conormal,
                subcell_mesh_velocity_on_faces, *box,
                typename DerivedCorrection::dg_package_data_volume_tags{},
                dg_package_data_projected_tags{});
            evolution::dg::Actions::detail::dg_package_data<System>(
                make_not_null(&lower_packaged_data),
                dynamic_cast<const DerivedCorrection&>(boundary_correction),
                vars_lower_face, lower_outward_conormal,
                subcell_mesh_velocity_on_faces, *box,
                typename DerivedCorrection::dg_package_data_volume_tags{},
                dg_package_data_projected_tags{});

            // Now need to check if any of our neighbors are doing DG,
            // because if so then we need to use whatever boundary data
            // they sent instead of what we computed locally.
            //
            // Note: We could check this beforehand to avoid the extra
            // work of reconstruction and flux computations at the
            // boundaries.
            evolution::dg::subcell::correct_package_data<true>(
                make_not_null(&lower_packaged_data),
                make_not_null(&upper_packaged_data), i, element, subcell_mesh,
                db::get<evolution::dg::Tags::MortarData<3>>(*box), 0);

            // Compute the corrections on the faces. We only need to
            // compute this once because we can just flip the normal
            // vectors then
            gsl::at(fd_boundary_corrections, i)
                .initialize(num_reconstructed_pts);
            evolution::dg::subcell::compute_boundary_terms(
                make_not_null(&gsl::at(fd_boundary_corrections, i)),
                dynamic_cast<const DerivedCorrection&>(boundary_correction),
                upper_packaged_data, lower_packaged_data);
            // We need to multiply by the normal vector normalization
            gsl::at(fd_boundary_corrections, i) *= get(normalization);
          }
        });

    std::array<double, 3> one_over_delta_xi{};
    {
      const tnsr::I<DataVector, 3, Frame::ElementLogical>&
          cell_centered_logical_coords =
              db::get<evolution::dg::subcell::Tags::Coordinates<
                  3, Frame::ElementLogical>>(*box);

      for (size_t i = 0; i < 3; ++i) {
        // Note: assumes isotropic extents
        gsl::at(one_over_delta_xi, i) =
            1.0 / (get<0>(cell_centered_logical_coords)[1] -
                   get<0>(cell_centered_logical_coords)[0]);
      }
    }

    // Now compute the actual time derivatives.
    using variables_tag = typename System::variables_tag;
    using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;

    const gsl::not_null<typename dt_variables_tag::type*> dt_vars_ptr =
        db::mutate<dt_variables_tag>(
            [](const auto local_dt_vars_ptr) { return local_dt_vars_ptr; },
            box);
    dt_vars_ptr->initialize(subcell_mesh.number_of_grid_points());

    using source_tags =
        tmpl::transform<ForceFree::Sources::return_tags,
                        tmpl::bind<db::remove_tag_prefix, tmpl::_1>>;
    using source_arg_tags = Sources::argument_tags;
    sources_impl(dt_vars_ptr, *box, source_tags{}, source_arg_tags{});

    // Zero the dt(U) for variables that do not have a source term. This is
    // necessary to avoid `+=` to a `NaN` (debug mode) or random garbage
    // (release mode) when adding mesh corrections below.
    tmpl::for_each<evolved_vars_tags>([&dt_vars_ptr](auto tag_v) {
      using tag = tmpl::type_from<decltype(tag_v)>;
      auto& dt_var = get<::Tags::dt<tag>>(*dt_vars_ptr);
      for (size_t i = 0; i < dt_var.size(); ++i) {
        if constexpr (not tmpl::list_contains_v<source_tags, tag>) {
          dt_var[i] = 0.0;
        }
      }
    });

    // Apply mesh velocity corrections to source terms if needed
    if (div_dg_mesh_velocity.has_value()) {
      const DataVector div_subcell_mesh_velocity =
          evolution::dg::subcell::fd::project(
              div_dg_mesh_velocity.value().get(), dg_mesh,
              subcell_mesh.extents());
      tmpl::for_each<evolved_vars_tags>(
          [&dt_vars_ptr, &div_subcell_mesh_velocity,
           &evolved_vars = db::get<variables_tag>(*box)](auto tag_v) {
            using tag = tmpl::type_from<decltype(tag_v)>;
            auto& dt_var = get<::Tags::dt<tag>>(*dt_vars_ptr);
            const auto& U = get<tag>(evolved_vars);
            for (size_t i = 0; i < dt_var.size(); ++i) {
              dt_var[i] -= div_subcell_mesh_velocity * U[i];
            }
          });
    }

    std::optional<std::array<Variables<evolved_vars_tags>, 3>>
        high_order_corrections{};
    ::fd::cartesian_high_order_flux_corrections(
        make_not_null(&high_order_corrections),
        db::get<evolution::dg::subcell::Tags::CellCenteredFlux<
            evolved_vars_tags, 3>>(*box),
        fd_boundary_corrections, fd_derivative_order,
        db::get<evolution::dg::subcell::Tags::GhostDataForReconstruction<3>>(
            *box),
        subcell_mesh, recons.ghost_zone_size(),
        reconstruction_order.value_or(
            std::array<gsl::span<std::uint8_t>, 3>{}));

    const auto& cell_centered_logical_to_grid_inv_jacobian = db::get<
        evolution::dg::subcell::fd::Tags::InverseJacobianLogicalToGrid<3>>(
        *box);
    for (size_t dim = 0; dim < 3; ++dim) {
      const auto& boundary_correction_in_axis =
          high_order_corrections.has_value()
              ? gsl::at(high_order_corrections.value(), dim)
              : gsl::at(fd_boundary_corrections, dim);
      const auto& component_inverse_jacobian =
          cell_centered_logical_to_grid_inv_jacobian.get(dim, dim);
      const double inverse_delta = gsl::at(one_over_delta_xi, dim);
      tmpl::for_each<typename variables_tag::tags_list>(
          [&dt_vars_ptr, &boundary_correction_in_axis,
           &component_inverse_jacobian, dim, inverse_delta,
           &subcell_mesh](auto evolved_var_tag_v) {
            using evolved_var_tag =
                tmpl::type_from<decltype(evolved_var_tag_v)>;
            using dt_tag = ::Tags::dt<evolved_var_tag>;
            auto& dt_var = get<dt_tag>(*dt_vars_ptr);
            const auto& var_correction =
                get<evolved_var_tag>(boundary_correction_in_axis);
            for (size_t i = 0; i < dt_var.size(); ++i) {
              evolution::dg::subcell::add_cartesian_flux_divergence(
                  make_not_null(&dt_var[i]), inverse_delta,
                  component_inverse_jacobian, var_correction[i],
                  subcell_mesh.extents(), dim);
            }
          });
    }

    evolution::dg::subcell::store_reconstruction_order_in_databox(
        box, reconstruction_order);
  }

 private:
  template <typename DtVarsList, typename DbTagsList, typename... SourcedTags,
            typename... ArgsTags>
  static void sources_impl(
      const gsl::not_null<Variables<DtVarsList>*> dt_vars_ptr,
      const db::DataBox<DbTagsList>& box, tmpl::list<SourcedTags...> /*meta*/,
      tmpl::list<ArgsTags...> /*meta*/) {
    ForceFree::Sources::apply(get<::Tags::dt<SourcedTags>>(dt_vars_ptr)...,
                              get<ArgsTags>(box)...);
  }
};

}  // namespace ForceFree::subcell
