// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/BoundaryConditions/NoIncomingPoynting.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/LeviCivitaIterator.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/SliceTensor.hpp"
#include "Evolution/Systems/ForceFree/ElectricCurrentDensity.hpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/ForceFree/Fluxes.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace ForceFree::BoundaryConditions {

NoIncomingPoynting::NoIncomingPoynting(CkMigrateMessage* const msg)
    : BoundaryCondition(msg) {}

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
NoIncomingPoynting::get_clone() const {
  return std::make_unique<NoIncomingPoynting>(*this);
}

void NoIncomingPoynting::pup(PUP::er& p) { BoundaryCondition::pup(p); }

// NOLINTNEXTLINE
PUP::able::PUP_ID NoIncomingPoynting::my_PUP_ID = 0;

std::optional<std::string> NoIncomingPoynting::dg_ghost(
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_e,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_b,
    const gsl::not_null<Scalar<DataVector>*> tilde_psi,
    const gsl::not_null<Scalar<DataVector>*> tilde_phi,
    const gsl::not_null<Scalar<DataVector>*> tilde_q,

    const gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*> tilde_e_flux,
    const gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*> tilde_b_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        tilde_psi_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        tilde_phi_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_q_flux,

    const gsl::not_null<Scalar<DataVector>*> lapse,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> shift,
    const gsl::not_null<tnsr::II<DataVector, 3, Frame::Inertial>*>
        inv_spatial_metric,

    const std::optional<
        tnsr::I<DataVector, 3, Frame::Inertial>>& /*face_mesh_velocity*/,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
    const tnsr::I<DataVector, 3, Frame::Inertial>& normal_vector,

    // interior evolved vars tags
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_tilde_e,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_tilde_b,
    const Scalar<DataVector>& interior_tilde_q,

    // interior temporary tags
    const Scalar<DataVector>& interior_lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_shift,
    const tnsr::II<DataVector, 3, Frame::Inertial>& interior_inv_spatial_metric,
    const double parallel_conductivity) {
  get(*lapse) = get(interior_lapse);
  for (size_t i = 0; i < 3; ++i) {
    (*shift).get(i) = interior_shift.get(i);
    for (size_t j = 0; j < 3; ++j) {
      (*inv_spatial_metric).get(i, j) = interior_inv_spatial_metric.get(i, j);
    }
  }

  const size_t number_of_grid_points = get(interior_lapse).size();
  Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
                       ::Tags::TempI<0, 3>, ::Tags::TempI<1, 3>,
                       //
                       TildeJ, SpatialMetric, SqrtDetSpatialMetric>>
      temp_buffer{number_of_grid_points, 0.0};

  // spatial metric and sqrt determinant of spatial metric can be retrived
  // from Databox but only as gridless_tags with whole volume data (unlike
  // all the other arguments which are face tensors). Rather than doing
  // expensive tensor slicing operations on those, we just compute those two
  // quantities from inverse spatial metric as below.
  auto& interior_spatial_metric = get<SpatialMetric>(temp_buffer);
  auto& interior_sqrt_det_spatial_metric =
      get<SqrtDetSpatialMetric>(temp_buffer);

  determinant_and_inverse(make_not_null(&interior_sqrt_det_spatial_metric),
                          make_not_null(&interior_spatial_metric),
                          interior_inv_spatial_metric);
  get(interior_sqrt_det_spatial_metric) =
      1.0 / sqrt(get(interior_sqrt_det_spatial_metric));

  // Compute the drift velocity v^i = (E x B)^i / B^2
  auto& drift_velocity = get<::Tags::TempI<0, 3>>(temp_buffer);
  for (LeviCivitaIterator<3> it; it; ++it) {
    const auto& i = it[0];
    const auto& j = it[1];
    const auto& k = it[2];
    drift_velocity.get(i) +=
        it.sign() * interior_tilde_e.get(j) * interior_tilde_b.get(k);
  }
  auto& tilde_b_squared = get<::Tags::TempScalar<0>>(temp_buffer);
  dot_product(make_not_null(&tilde_b_squared), interior_tilde_b,
              interior_tilde_b, interior_spatial_metric);
  for (size_t i = 0; i < 3; ++i) {
    drift_velocity.get(i) /= get(tilde_b_squared);
  }

  // and its product with normal vector (v^i n_i)
  auto& normal_dot_drift_velocity = get<::Tags::TempScalar<1>>(temp_buffer);
  dot_product(make_not_null(&normal_dot_drift_velocity), drift_velocity,
              normal_covector);

  // Now compute the exterior state
  for (size_t i = 0; i < 3; ++i) {
    tilde_e->get(i) = interior_tilde_e.get(i);
    tilde_b->get(i) = interior_tilde_b.get(i);
  }
  get(*tilde_psi) = 0.0;
  get(*tilde_phi) = 0.0;
  get(*tilde_q) = get(interior_tilde_q);

  // Fix the electric field when (v_i n^i) < 0
  for (size_t m = 0; m < number_of_grid_points; ++m) {
    if (get(normal_dot_drift_velocity)[m] < 0.0) {
      for (size_t d = 0; d < 3; ++d) {
        // Project out normal component of drift velocity
        drift_velocity.get(d)[m] -=
            get(normal_dot_drift_velocity)[m] * normal_vector.get(d)[m];
        (*tilde_e).get(d)[m] = 0.0;
      }

      // E = B x v
      for (LeviCivitaIterator<3> it; it; ++it) {
        const auto& i = it[0];
        const auto& j = it[1];
        const auto& k = it[2];
        (*tilde_e).get(i)[m] +=
            it.sign() * interior_tilde_b.get(j)[m] * drift_velocity.get(k)[m];
      }
    }
  }

  auto& exterior_tilde_j = get<::Tags::TempI<1, 3>>(temp_buffer);
  ForceFree::ComputeDriftTildeJ::apply(
      make_not_null(&exterior_tilde_j), *tilde_q, *tilde_e, *tilde_b,
      parallel_conductivity, *lapse, interior_sqrt_det_spatial_metric,
      interior_spatial_metric, std::optional<Scalar<DataVector>>{});

  Fluxes::apply(tilde_e_flux, tilde_b_flux, tilde_psi_flux, tilde_phi_flux,
                tilde_q_flux, *tilde_e, *tilde_b, *tilde_psi, *tilde_phi,
                *tilde_q, exterior_tilde_j, interior_lapse, interior_shift,
                interior_sqrt_det_spatial_metric, interior_spatial_metric,
                interior_inv_spatial_metric);

  return {};
}

void NoIncomingPoynting::fd_ghost(
    gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_j,
    gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_e,
    gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_b,
    gsl::not_null<Scalar<DataVector>*> tilde_psi,
    gsl::not_null<Scalar<DataVector>*> tilde_phi,
    gsl::not_null<Scalar<DataVector>*> tilde_q,

    gsl::not_null<std::optional<Variables<
        db::wrap_tags_in<Flux, typename ForceFree::System::flux_variables>>>*>
        cell_centered_ghost_fluxes,

    const Direction<3>& direction,

    // fd_interior_evolved_variables_tags
    const tnsr::I<DataVector, 3, Frame::Inertial>& volume_tilde_e,
    const tnsr::I<DataVector, 3, Frame::Inertial>& volume_tilde_b,
    const Scalar<DataVector>& volume_tilde_q,

    // fd_interior_temporary_tags
    const tnsr::I<DataVector, 3, Frame::Inertial>& volume_tilde_j,
    const Scalar<DataVector>& volume_lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& volume_shift,
    const tnsr::II<DataVector, 3, Frame::Inertial>& volume_inv_spatial_metric,
    const ::InverseJacobian<DataVector, 3, Frame::ElementLogical,
                            Frame::Inertial>& inv_jacobian_dg,
    const Mesh<3>& dg_mesh, const Mesh<3>& subcell_mesh,

    // fd_gridless_tags
    double parallel_conductivity,
    const fd::Reconstructor& reconstructor) const {
  const size_t direction_dim{direction.dimension()};

  const auto subcell_extents{subcell_mesh.extents()};
  const size_t num_subcell_face_pts{
      subcell_extents.slice_away(direction_dim).product()};

  // The non-reflective (transparent) outer state at the innermost ghost zoneu
  // and temporary quantities for computing it.
  Variables<tmpl::list<TildeJ, TildeE, TildeB, TildeQ, Lapse, Shift,
                       SqrtDetSpatialMetric, SpatialMetric, InvSpatialMetric>>
      innermost_ghost_vars{num_subcell_face_pts, 0.0};

  // Metric into ghost vars
  auto& interior_lapse = get<Lapse>(innermost_ghost_vars);
  auto& interior_shift = get<Shift>(innermost_ghost_vars);
  auto& interior_sqrt_det_spatial_metric =
      get<SqrtDetSpatialMetric>(innermost_ghost_vars);
  auto& interior_inv_spatial_metric =
      get<InvSpatialMetric>(innermost_ghost_vars);
  auto& interior_spatial_metric = get<SpatialMetric>(innermost_ghost_vars);

  evolution::dg::subcell::slice_tensor_for_subcell(
      make_not_null(&interior_lapse), volume_lapse, subcell_extents, 1,
      direction, {});
  evolution::dg::subcell::slice_tensor_for_subcell(
      make_not_null(&interior_shift), volume_shift, subcell_extents, 1,
      direction, {});
  evolution::dg::subcell::slice_tensor_for_subcell(
      make_not_null(&interior_inv_spatial_metric), volume_inv_spatial_metric,
      subcell_extents, 1, direction, {});
  determinant_and_inverse(make_not_null(&interior_sqrt_det_spatial_metric),
                          make_not_null(&interior_spatial_metric),
                          interior_inv_spatial_metric);
  get(interior_sqrt_det_spatial_metric) =
      1.0 / sqrt(get(interior_sqrt_det_spatial_metric));

  // =========================================================================
  // Temp buffer contains sliced E, B, J, sliced metric, normal vectors, and all
  // the temporaries..
  Variables<
      tmpl::list<TildeE, TildeB, TildeQ, TildeJ,
                 // For storing normal (co)vector on subcell face mesh
                 ::Tags::Tempi<0, 3>, ::Tags::TempI<0, 3>,
                 // For temporary quantities
                 ::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
                 ::Tags::TempScalar<2>, ::Tags::TempI<1, 3>,
                 ::Tags::TempI<2, 3>, ::Tags::TempI<3, 3>, ::Tags::TempI<4, 3>>>
      slice_buffer{num_subcell_face_pts, 0.0};

  auto& interior_tilde_e = get<TildeE>(slice_buffer);
  auto& interior_tilde_b = get<TildeB>(slice_buffer);
  auto& interior_tilde_q = get<TildeQ>(slice_buffer);
  auto& interior_tilde_j = get<TildeJ>(slice_buffer);

  // Slice and store outermost values of volume tensors to the buffer
  evolution::dg::subcell::slice_tensor_for_subcell(
      make_not_null(&interior_tilde_e), volume_tilde_e, subcell_extents, 1,
      direction, {});
  evolution::dg::subcell::slice_tensor_for_subcell(
      make_not_null(&interior_tilde_b), volume_tilde_b, subcell_extents, 1,
      direction, {});
  evolution::dg::subcell::slice_tensor_for_subcell(
      make_not_null(&interior_tilde_q), volume_tilde_q, subcell_extents, 1,
      direction, {});
  evolution::dg::subcell::slice_tensor_for_subcell(
      make_not_null(&interior_tilde_j), volume_tilde_j, subcell_extents, 1,
      direction, {});

  // Construct normal vector and normal covector on the subcell face mesh.
  // Project normal covector from DG to FD first, then compute normal vector on
  // FD mesh using inverse spatial metric
  auto& subcell_normal_covector = get<::Tags::Tempi<0, 3>>(slice_buffer);
  auto& subcell_normal_vector = get<::Tags::TempI<0, 3>>(slice_buffer);
  {
    // The unnormalized normal vector is n_j = d \xi^{\hat i}/dx^j
    // with "i" the current face.
    tnsr::i<DataVector, 3, Frame::Inertial> subcell_volume_normal_covector{
        subcell_extents.product(), 0.0};
    for (size_t j = 0; j < 3; ++j) {
      subcell_volume_normal_covector.get(j) =
          evolution::dg::subcell::fd::project_to_faces(
              inv_jacobian_dg.get(direction_dim, j), dg_mesh, subcell_extents,
              direction_dim);
    }
    evolution::dg::subcell::slice_tensor_for_subcell(
        make_not_null(&subcell_normal_covector), subcell_volume_normal_covector,
        subcell_extents, 1, direction, {});

    // Apply normalization to covector and compute normal vector n^j
    const Scalar<DataVector> normalization{
        sqrt(get(dot_product(subcell_normal_covector, subcell_normal_covector,
                             interior_inv_spatial_metric)))};
    for (size_t j = 0; j < 3; j++) {
      subcell_normal_covector.get(j) *= direction.sign() / get(normalization);
    }
    raise_or_lower_index(make_not_null(&subcell_normal_vector),
                         subcell_normal_covector, interior_inv_spatial_metric);
  }

  // Now we compute the actual ghost state
  auto& exterior_tilde_j = get<TildeJ>(innermost_ghost_vars);
  auto& exterior_tilde_e = get<TildeE>(innermost_ghost_vars);
  auto& exterior_tilde_b = get<TildeB>(innermost_ghost_vars);
  auto& exterior_tilde_q = get<TildeQ>(innermost_ghost_vars);

  // Compute the drift velocity
  auto& drift_velocity = get<::Tags::TempI<1, 3>>(slice_buffer);
  for (LeviCivitaIterator<3> it; it; ++it) {
    const auto& i = it[0];
    const auto& j = it[1];
    const auto& k = it[2];
    drift_velocity.get(i) +=
        it.sign() * interior_tilde_e.get(j) * interior_tilde_b.get(k);
  }
  auto& tilde_b_squared = get<::Tags::TempScalar<0>>(slice_buffer);
  dot_product(make_not_null(&tilde_b_squared), interior_tilde_b,
              interior_tilde_b, interior_spatial_metric);
  for (size_t i = 0; i < 3; ++i) {
    drift_velocity.get(i) /= get(tilde_b_squared);
  }

  // and  n.v_d
  auto& normal_dot_drift_velocity = get<::Tags::TempScalar<1>>(slice_buffer);
  dot_product(make_not_null(&normal_dot_drift_velocity), drift_velocity,
              subcell_normal_covector);

  // Compute the ghost state

  for (size_t i = 0; i < 3; ++i) {
    exterior_tilde_e.get(i) = interior_tilde_e.get(i);
    exterior_tilde_b.get(i) = interior_tilde_b.get(i);
  }
  get(exterior_tilde_q) = get(interior_tilde_q);

  // Compute the filtered electric field
  for (size_t m = 0; m < num_subcell_face_pts; ++m) {
    if (get(normal_dot_drift_velocity)[m] < 0.0) {
      // Project out normal component of drift velocity
      for (size_t d = 0; d < 3; ++d) {
        drift_velocity.get(d)[m] -=
            get(normal_dot_drift_velocity)[m] * subcell_normal_vector.get(d)[m];
        exterior_tilde_e.get(d)[m] = 0.0;
      }
      // E = B x Vd
      for (LeviCivitaIterator<3> it; it; ++it) {
        const auto& i = it[0];
        const auto& j = it[1];
        const auto& k = it[2];
        exterior_tilde_e.get(i)[m] +=
            it.sign() * interior_tilde_b.get(j)[m] * drift_velocity.get(k)[m];
      }
    }
  }

  ForceFree::ComputeDriftTildeJ::apply(
      make_not_null(&exterior_tilde_j), exterior_tilde_q, exterior_tilde_e,
      exterior_tilde_b, parallel_conductivity, interior_lapse,
      interior_sqrt_det_spatial_metric, interior_spatial_metric,
      std::optional<Scalar<DataVector>>{});

  // Copy
  const size_t ghost_zone_size = reconstructor.ghost_zone_size();
  Variables<tmpl::list<TildeJ, TildeE, TildeB, TildeQ, Lapse, Shift,
                       SqrtDetSpatialMetric, SpatialMetric, InvSpatialMetric>>
      fd_ghost_vars{num_subcell_face_pts * ghost_zone_size, 0.0};
  Index<3> ghost_data_extents = subcell_extents;
  ghost_data_extents[direction_dim] = ghost_zone_size;

  for (size_t i_ghost = 0; i_ghost < ghost_zone_size; ++i_ghost) {
    add_slice_to_data(make_not_null(&fd_ghost_vars), innermost_ghost_vars,
                      ghost_data_extents, direction_dim, i_ghost);
  }

  for (size_t i = 0; i < 3; ++i) {
    (*tilde_j).get(i) = get<TildeJ>(fd_ghost_vars).get(i);

    (*tilde_e).get(i) = get<TildeE>(fd_ghost_vars).get(i);
    (*tilde_b).get(i) = get<TildeB>(fd_ghost_vars).get(i);
  }
  get(*tilde_q) = get(get<TildeQ>(fd_ghost_vars));

  get(*tilde_psi) = 0.0;
  get(*tilde_phi) = 0.0;

  if (cell_centered_ghost_fluxes->has_value()) {
    Fluxes::apply(
        make_not_null(
            &get<Flux<Tags::TildeE>>(cell_centered_ghost_fluxes->value())),
        make_not_null(
            &get<Flux<Tags::TildeB>>(cell_centered_ghost_fluxes->value())),
        make_not_null(
            &get<Flux<Tags::TildePsi>>(cell_centered_ghost_fluxes->value())),
        make_not_null(
            &get<Flux<Tags::TildePhi>>(cell_centered_ghost_fluxes->value())),
        make_not_null(
            &get<Flux<Tags::TildeQ>>(cell_centered_ghost_fluxes->value())),
        *tilde_e, *tilde_b, *tilde_psi, *tilde_phi, *tilde_q, *tilde_j,
        get<Lapse>(fd_ghost_vars), get<Shift>(fd_ghost_vars),
        get<SqrtDetSpatialMetric>(fd_ghost_vars),
        get<SpatialMetric>(fd_ghost_vars),
        get<InvSpatialMetric>(fd_ghost_vars));
  }
}

}  // namespace ForceFree::BoundaryConditions
