// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DgSubcell/CartesianFluxDivergence.hpp"
#include "Evolution/DgSubcell/ComputeBoundaryTerms.hpp"
#include "Evolution/DgSubcell/CorrectPackagedData.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/Jacobians.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/OnSubcellFaces.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/PackageDataImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/Systems/ForceFree/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/ForceFree/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/ForceFree/Fluxes.hpp"
#include "Evolution/Systems/ForceFree/Sources.hpp"
#include "Evolution/Systems/ForceFree/Subcell/ComputeFluxes.hpp"
#include "Evolution/Systems/ForceFree/System.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
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
 * - Assumes the mesh is not moving (grid and inertial frame are the same)
 */
struct TimeDerivative {
  template <typename DbTagsList>
  static void apply(const gsl::not_null<db::DataBox<DbTagsList>*> box) {
    // subcell is currently not supported for external boundary elements
    const Element<3>& element = db::get<domain::Tags::Element<3>>(*box);
    ASSERT(element.external_boundaries().size() == 0,
           "Can't have external boundaries right now with subcell. ElementID "
               << element.id());

    using evolved_vars_tags = typename System::variables_tag::tags_list;
    using fluxes_tags = typename Fluxes::return_tags;

    // The copy of Mesh is intentional to avoid a GCC-7 internal compiler error.
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

    // boundary correction terms of evolved variables on subcell interfaces
    std::array<Variables<evolved_vars_tags>, 3> fd_boundary_corrections{};

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
          call_with_dynamic_type<
              void, typename ForceFree::fd::Reconstructor::creatable_classes>(
              &recons, [&box, &package_data_argvars_lower_face_with_tildej,
                        &package_data_argvars_upper_face_with_tildej](
                           const auto& reconstructor) {
                db::apply<typename std::decay_t<
                    decltype(*reconstructor)>::reconstruction_argument_tags>(
                    [&package_data_argvars_lower_face_with_tildej,
                     &package_data_argvars_upper_face_with_tildej,
                     &reconstructor](const auto&... args) {
                      reconstructor->reconstruct(
                          make_not_null(
                              &package_data_argvars_lower_face_with_tildej),
                          make_not_null(
                              &package_data_argvars_upper_face_with_tildej),
                          args...);
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
                vars_upper_face, upper_outward_conormal, {std::nullopt}, *box,
                typename DerivedCorrection::dg_package_data_volume_tags{},
                dg_package_data_projected_tags{});
            evolution::dg::Actions::detail::dg_package_data<System>(
                make_not_null(&lower_packaged_data),
                dynamic_cast<const DerivedCorrection&>(boundary_correction),
                vars_lower_face, lower_outward_conormal, {std::nullopt}, *box,
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
    using dt_variables_tag =
        db::add_tag_prefix<::Tags::dt, typename System::variables_tag>;
    using source_arg_tags = Sources::argument_tags;

    const size_t num_grid_pts = subcell_mesh.number_of_grid_points();

    db::mutate_apply<tmpl::list<dt_variables_tag>, source_arg_tags>(
        [&num_grid_pts, &fd_boundary_corrections, &subcell_mesh,
         &one_over_delta_xi,
         &cell_centered_logical_to_grid_inv_jacobian = db::get<
             evolution::dg::subcell::fd::Tags::InverseJacobianLogicalToGrid<3>>(
             *box)](const auto dt_vars_ptr, const auto&... source_args) {
          dt_vars_ptr->initialize(num_grid_pts, 0.0);

          auto& dt_tilde_e =
              get<::Tags::dt<ForceFree::Tags::TildeE>>(*dt_vars_ptr);
          auto& dt_tilde_b =
              get<::Tags::dt<ForceFree::Tags::TildeB>>(*dt_vars_ptr);
          auto& dt_tilde_psi =
              get<::Tags::dt<ForceFree::Tags::TildePsi>>(*dt_vars_ptr);
          auto& dt_tilde_phi =
              get<::Tags::dt<ForceFree::Tags::TildePhi>>(*dt_vars_ptr);
          auto& dt_tilde_q =
              get<::Tags::dt<ForceFree::Tags::TildeQ>>(*dt_vars_ptr);

          Sources::apply(make_not_null(&dt_tilde_e), make_not_null(&dt_tilde_b),
                         make_not_null(&dt_tilde_psi),
                         make_not_null(&dt_tilde_phi), source_args...);

          for (size_t dim = 0; dim < 3; ++dim) {
            tnsr::I<DataVector, 3, Frame::Inertial>&
                tilde_e_density_correction = get<ForceFree::Tags::TildeE>(
                    gsl::at(fd_boundary_corrections, dim));
            tnsr::I<DataVector, 3, Frame::Inertial>&
                tilde_b_density_correction = get<ForceFree::Tags::TildeB>(
                    gsl::at(fd_boundary_corrections, dim));
            Scalar<DataVector>& tilde_psi_density_correction =
                get<ForceFree::Tags::TildePsi>(
                    gsl::at(fd_boundary_corrections, dim));
            Scalar<DataVector>& tilde_phi_density_correction =
                get<ForceFree::Tags::TildePhi>(
                    gsl::at(fd_boundary_corrections, dim));
            Scalar<DataVector>& tilde_q_density_correction =
                get<ForceFree::Tags::TildeQ>(
                    gsl::at(fd_boundary_corrections, dim));

            for (size_t d = 0; d < 3; ++d) {
              evolution::dg::subcell::add_cartesian_flux_divergence(
                  make_not_null(&dt_tilde_e.get(d)),
                  gsl::at(one_over_delta_xi, dim),
                  cell_centered_logical_to_grid_inv_jacobian.get(dim, dim),
                  tilde_e_density_correction.get(d), subcell_mesh.extents(),
                  dim);
              evolution::dg::subcell::add_cartesian_flux_divergence(
                  make_not_null(&dt_tilde_b.get(d)),
                  gsl::at(one_over_delta_xi, dim),
                  cell_centered_logical_to_grid_inv_jacobian.get(dim, dim),
                  tilde_b_density_correction.get(d), subcell_mesh.extents(),
                  dim);
            }
            evolution::dg::subcell::add_cartesian_flux_divergence(
                make_not_null(&get(dt_tilde_psi)),
                gsl::at(one_over_delta_xi, dim),
                cell_centered_logical_to_grid_inv_jacobian.get(dim, dim),
                get(tilde_psi_density_correction), subcell_mesh.extents(), dim);
            evolution::dg::subcell::add_cartesian_flux_divergence(
                make_not_null(&get(dt_tilde_phi)),
                gsl::at(one_over_delta_xi, dim),
                cell_centered_logical_to_grid_inv_jacobian.get(dim, dim),
                get(tilde_phi_density_correction), subcell_mesh.extents(), dim);
            evolution::dg::subcell::add_cartesian_flux_divergence(
                make_not_null(&get(dt_tilde_q)),
                gsl::at(one_over_delta_xi, dim),
                cell_centered_logical_to_grid_inv_jacobian.get(dim, dim),
                get(tilde_q_density_correction), subcell_mesh.extents(), dim);
          }
        },
        box);
  }
};

}  // namespace ForceFree::subcell
