// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/BoundaryConditions/DemandOutgoingCharSpeeds.hpp"

#include <limits>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/SliceTensor.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/TMPL.hpp"

#include <iostream>

namespace ForceFree::BoundaryConditions {
DemandOutgoingCharSpeeds::DemandOutgoingCharSpeeds(CkMigrateMessage* const msg)
    : BoundaryCondition(msg) {}

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
DemandOutgoingCharSpeeds::get_clone() const {
  return std::make_unique<DemandOutgoingCharSpeeds>(*this);
}

void DemandOutgoingCharSpeeds::pup(PUP::er& p) { BoundaryCondition::pup(p); }

// NOLINTNEXTLINE
PUP::able::PUP_ID DemandOutgoingCharSpeeds::my_PUP_ID = 0;

std::optional<std::string>
DemandOutgoingCharSpeeds::dg_demand_outgoing_char_speeds(
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
        face_mesh_velocity,
    const tnsr::i<DataVector, 3, Frame::Inertial>&
        outward_directed_normal_covector,
    const tnsr::I<DataVector, 3, Frame::Inertial>&
    /*outward_directed_normal_vector*/,

    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const Scalar<DataVector>& lapse) {
  double min_speed = std::numeric_limits<double>::signaling_NaN();

  Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>>> buffer{
      get(lapse).size()};

  auto& normal_dot_shift = get<::Tags::TempScalar<0>>(buffer);
  dot_product(make_not_null(&normal_dot_shift),
              outward_directed_normal_covector, shift);

  if (face_mesh_velocity.has_value()) {
    auto& normal_dot_mesh_velocity = get<::Tags::TempScalar<1>>(buffer);
    dot_product(make_not_null(&normal_dot_mesh_velocity),
                outward_directed_normal_covector, face_mesh_velocity.value());
    get(normal_dot_shift) += get(normal_dot_mesh_velocity);
  }

  // The characteristic speeds are bounded by \pm \alpha - \beta^i n_i,
  // therefore minimum is given as `-\alpha - \beta^i n_i`.
  min_speed = min(-get(lapse) - get(normal_dot_shift));

  if (min_speed < 0.0) {
    return {MakeString{}
            << "DemandOutgoingCharSpeeds boundary condition violated. Speed: "
            << min_speed << "\nn_i: " << outward_directed_normal_covector
            << "\n"};
  }

  return std::nullopt;
}

void DemandOutgoingCharSpeeds::fd_demand_outgoing_char_speeds(
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_j,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_e,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_b,
    const gsl::not_null<Scalar<DataVector>*> tilde_psi,
    const gsl::not_null<Scalar<DataVector>*> tilde_phi,
    const gsl::not_null<Scalar<DataVector>*> tilde_q,

    const gsl::not_null<std::optional<Variables<
        db::wrap_tags_in<Flux, typename ForceFree::System::flux_variables>>>*>
        cell_centered_ghost_fluxes,

    const Direction<3>& direction,
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
        dg_volume_mesh_velocity,

    // fd_interior_evolved_variables_tags
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_tilde_e,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_tilde_b,
    const Scalar<DataVector>& interior_tilde_psi,
    const Scalar<DataVector>& interior_tilde_phi,
    const Scalar<DataVector>& interior_tilde_q,

    // fd_interior_temporary_tags
    const Scalar<DataVector>& volume_lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& volume_shift,
    const tnsr::II<DataVector, 3, Frame::Inertial>& volume_inv_spatial_metric,
    const ::InverseJacobian<DataVector, 3, Frame::ElementLogical,
                            Frame::Inertial>& inv_jacobian_dg,
    const Mesh<3>& dg_mesh, const Mesh<3>& subcell_mesh,

    // fd_gridless_tags
    const double parallel_conductivity,
    const fd::Reconstructor& reconstructor) {
  double min_char_speed = std::numeric_limits<double>::signaling_NaN();

  const size_t ghost_zone_size{reconstructor.ghost_zone_size()};
  const size_t dim_direction{direction.dimension()};

  const auto subcell_extents{subcell_mesh.extents()};
  const size_t num_subcell_face_pts{
      subcell_extents.slice_away(dim_direction).product()};

  // The boundary condition below simply uses the outermost values on
  // cell-centered FD grid points to estimate face values on the external
  // boundary. This is equivalent to adopting the piecewise constant (lowest
  // order) FD reconstruction for FD cells at the external boundaries.
  //
  // In the future we may want to use more accurate methods (for instance,
  // one-sided characteristic reconstruction using WENO) for imposing
  // higher-order DemandOutgoingCharSpeeds boundary condition.

  // We need three Variables objects to store:
  //  1. Reconstructed variables at the innermost ghost zone (a single slice)
  //  2. Reconstructed variables at the all ghost zone (multiple slices with
  //     depth being `ghost_zone_size`)
  //  3. Temporary quantities (e.g. subcell normal vector, normal_dot_shift ..)
  //
  // So allocate a giant buffer to store all of these
  Variables<tmpl::list<
      // Metric quantities
      Lapse, Shift, InvSpatialMetric,
      // Temporaries
      ::Tags::TempScalar<0>, ::Tags::Tempi<0, 3>, ::Tags::TempI<0, 3>>>
      temporary_vars{num_subcell_face_pts, 0.0};
  Variables<tmpl::list<TildeE, TildeB, TildePsi>> innermost_ghost_vars{
      num_subcell_face_pts, 0.0};
  Variables<tmpl::list<TildeE, TildeB, TildePsi>> all_ghost_vars{
      num_subcell_face_pts * ghost_zone_size, 0.0};

  auto& lapse_at_boundary = get<Lapse>(temporary_vars);
  auto& shift_at_boundary = get<Shift>(temporary_vars);
  auto& inv_spatial_metric_at_boundary = get<InvSpatialMetric>(temporary_vars);
  evolution::dg::subcell::slice_tensor_for_subcell(
      make_not_null(&lapse_at_boundary), volume_lapse, subcell_extents, 1,
      direction, {});
  evolution::dg::subcell::slice_tensor_for_subcell(
      make_not_null(&shift_at_boundary), volume_shift, subcell_extents, 1,
      direction, {});
  evolution::dg::subcell::slice_tensor_for_subcell(
      make_not_null(&inv_spatial_metric_at_boundary), volume_inv_spatial_metric,
      subcell_extents, 1, direction, {});

  // Construct normal vector and normal covector on the subcell face mesh.
  // Project normal covector from DG to FD first, then compute normal vector on
  // FD mesh using inverse spatial metric
  auto& subcell_face_normal_covector = get<::Tags::Tempi<0, 3>>(temporary_vars);
  {
    // The unnormalized normal vector is n_j = d \xi^{\hat i}/dx^j
    // with "i" the current face.
    tnsr::i<DataVector, 3, Frame::Inertial> subcell_volume_normal_covector{
        subcell_extents.product(), 0.0};
    for (size_t j = 0; j < 3; ++j) {
      subcell_volume_normal_covector.get(j) =
          evolution::dg::subcell::fd::project_to_faces(
              inv_jacobian_dg.get(dim_direction, j), dg_mesh, subcell_extents,
              dim_direction);
    }
    evolution::dg::subcell::slice_tensor_for_subcell(
        make_not_null(&subcell_face_normal_covector),
        subcell_volume_normal_covector, subcell_extents, 1, direction, {});

    // Apply normalization to covector and compute normal vector n^j
    const Scalar<DataVector> normalization{sqrt(get(
        dot_product(subcell_face_normal_covector, subcell_face_normal_covector,
                    inv_spatial_metric_at_boundary)))};
    for (size_t j = 0; j < 3; j++) {
      subcell_face_normal_covector.get(j) *=
          direction.sign() / get(normalization);
    }
  }

  // Check the outgoing characteristics condition
  auto& normal_dot_shift = get<::Tags::TempScalar<0>>(temporary_vars);
  dot_product(make_not_null(&normal_dot_shift), subcell_face_normal_covector,
              shift_at_boundary);

  if (dg_volume_mesh_velocity.has_value()) {
    // We need to compute subcell mesh velocity at the boundary.
    // Project the DG mesh velocity onto subcell grid, then slice the outermost
    // one.
    tnsr::I<DataVector, 3, Frame::Inertial> subcell_volume_mesh_velocity{
        subcell_extents.product(), 0.0};
    for (size_t j = 0; j < 3; ++j) {
      subcell_volume_mesh_velocity.get(j) =
          evolution::dg::subcell::fd::project_to_faces(
              inv_jacobian_dg.get(dim_direction, j), dg_mesh, subcell_extents,
              dim_direction);
    }

    auto& subcell_face_mesh_velocity = get<::Tags::TempI<0, 3>>(temporary_vars);
    evolution::dg::subcell::slice_tensor_for_subcell(
        make_not_null(&subcell_face_mesh_velocity),
        subcell_volume_mesh_velocity, subcell_extents, 1, direction, {});

    auto& normal_dot_mesh_velocity = get<::Tags::TempScalar<0>>(temporary_vars);
    dot_product(make_not_null(&normal_dot_mesh_velocity),
                subcell_face_normal_covector, subcell_face_mesh_velocity);
    get(normal_dot_shift) += get(normal_dot_mesh_velocity);
  }

  // The characteristic speeds are bounded by \pm \alpha - \beta^i n_i,
  // therefore minimum is given as `-\alpha - \beta^i n_i`.
  min_char_speed = min(-get(lapse_at_boundary) - get(normal_dot_shift));

  if (min_char_speed < 0.0) {
    ERROR(
        "Subcell DemandOutgoingCharSpeeds boundary condition violated. Speed: "
        << min_char_speed << "\nn_i: " << subcell_face_normal_covector << "\n");
  } else {
    // Once the DemandOutgoingCharSpeeds condition has been checked, we fill
    // each slices of the ghost data with the boundary values. The reason that
    // we need this step is to prevent floating point exceptions being raised
    // while computing the subcell time derivative because of NaN or
    // uninitialized values in ghost data.

    evolution::dg::subcell::slice_tensor_for_subcell(
        make_not_null(&get<TildeE>(innermost_ghost_vars)), interior_tilde_e,
        subcell_extents, 1, direction, {});
    evolution::dg::subcell::slice_tensor_for_subcell(
        make_not_null(&get<TildeB>(innermost_ghost_vars)), interior_tilde_b,
        subcell_extents, 1, direction, {});
    evolution::dg::subcell::slice_tensor_for_subcell(
        make_not_null(&get<TildePsi>(innermost_ghost_vars)), interior_tilde_psi,
        subcell_extents, 1, direction, {});

    Index<3> ghost_data_extents = subcell_extents;
    ghost_data_extents[dim_direction] = ghost_zone_size;
    for (size_t i_ghost = 0; i_ghost < ghost_zone_size; ++i_ghost) {
      add_slice_to_data(make_not_null(&all_ghost_vars), innermost_ghost_vars,
                        ghost_data_extents, dim_direction, i_ghost);
    }

    // Copy the ghost data into
    for (size_t i = 0; i < 3; ++i) {
      tilde_e->get(i) = get<TildeE>(all_ghost_vars).get(i);
      tilde_b->get(i) = get<TildeB>(all_ghost_vars).get(i);

      tilde_j->get(i) = 0.0;  // FIXME : does this work?
    }

    get(*tilde_psi) = 0.0;
    get(*tilde_phi) = 0.0;
    get(*tilde_q) = 0.0;
  }

  // std::cout << "\n------------------------- \n * Direction : " << direction
  //           << std::endl;
  // std::cout << "\n Ghost Vars : " << all_ghost_vars << std::endl;

  // std::cout << "\n Outermost TildeE : " << get<TildeE>(innermost_ghost_vars)
  //           << std::endl;

  // std::cout << "\n TildeE = " << (*tilde_e) << std::endl;
}

}  // namespace ForceFree::BoundaryConditions
