// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/BoundaryConditions/Nonreflecting.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/ForceFree/ElectricCurrentDensity.hpp"
#include "Evolution/Systems/ForceFree/Fluxes.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"

#include "DataStructures/LeviCivitaIterator.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/SliceTensor.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"

#include <iostream>

namespace ForceFree::BoundaryConditions {

Nonreflecting::Nonreflecting(CkMigrateMessage* const msg)
    : BoundaryCondition(msg) {}

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
Nonreflecting::get_clone() const {
  return std::make_unique<Nonreflecting>(*this);
}

void Nonreflecting::pup(PUP::er& p) { BoundaryCondition::pup(p); }

// NOLINTNEXTLINE
PUP::able::PUP_ID Nonreflecting::my_PUP_ID = 0;

std::optional<std::string> Nonreflecting::dg_ghost(
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
  Variables<
      tmpl::list<::Tags::TempI<0, 3>, ::Tags::TempI<1, 3>, ::Tags::TempI<2, 3>,
                 ::Tags::TempI<3, 3>, ::Tags::TempScalar<0>,
                 ::Tags::TempScalar<1>, ::Tags::TempScalar<2>, Tags::TildeJ,
                 gr::Tags::SpatialMetric<DataVector, 3>,
                 gr::Tags::SqrtDetSpatialMetric<DataVector>>>
      temp_buffer{number_of_grid_points, 0.0};

  auto& normal_dot_tilde_e = get<::Tags::TempScalar<0>>(temp_buffer);
  dot_product(make_not_null(&normal_dot_tilde_e), interior_tilde_e,
              normal_covector);

  // Polarized EM modes
  auto& normal_cross_tilde_b = get<::Tags::TempI<0, 3>>(temp_buffer);
  for (LeviCivitaIterator<3> it; it; ++it) {
    const auto& i = it[0];
    const auto& j = it[1];
    const auto& k = it[2];
    normal_cross_tilde_b.get(i) +=
        it.sign() * normal_vector.get(j) * interior_tilde_b.get(k);
  }
  auto& tilde_e_projected = get<::Tags::TempI<1, 3>>(temp_buffer);
  for (size_t d = 0; d < 3; ++d) {
    tilde_e_projected.get(d) = interior_tilde_e.get(d) -
                               get(normal_dot_tilde_e) * normal_vector.get(d);
  }
  // Compute PE - n x B
  auto& outgoing_char_vector = get<::Tags::TempI<2, 3>>(temp_buffer);
  for (size_t d = 0; d < 3; ++d) {
    outgoing_char_vector.get(d) =
        tilde_e_projected.get(d) - normal_cross_tilde_b.get(d);
    (*tilde_e).get(d) = 0.5 * outgoing_char_vector.get(d);
    (*tilde_b).get(d) = 0.0;
  }
  for (LeviCivitaIterator<3> it; it; ++it) {
    const auto& i = it[0];
    const auto& j = it[1];
    const auto& k = it[2];
    (*tilde_b).get(i) +=
        it.sign() * 0.5 * normal_vector.get(j) * outgoing_char_vector.get(k);
  }

  // Unphysical modes associated with div cleaning fields :
  // Just copy the normal component, ignore div cleaning fields.
  auto& normal_dot_tilde_b = get<::Tags::TempScalar<1>>(temp_buffer);
  dot_product(make_not_null(&normal_dot_tilde_b), interior_tilde_b,
              normal_covector);
  for (size_t d = 0; d < 3; ++d) {
    (*tilde_e).get(d) += get(normal_dot_tilde_e) * normal_vector.get(d);
    (*tilde_b).get(d) += get(normal_dot_tilde_b) * normal_vector.get(d);
  }
  get(*tilde_psi) = 0.0;
  get(*tilde_phi) = 0.0;

  // Charge drift mode
  auto& tilde_e_cross_tilde_b = get<::Tags::TempI<3, 3>>(temp_buffer);
  for (LeviCivitaIterator<3> it; it; ++it) {
    const auto& i = it[0];
    const auto& j = it[1];
    const auto& k = it[2];
    tilde_e_cross_tilde_b.get(i) +=
        it.sign() * interior_tilde_e.get(j) * interior_tilde_b.get(k);
  }
  auto& normal_dot_tilde_e_cross_tilde_b =
      get<::Tags::TempScalar<2>>(temp_buffer);
  dot_product(make_not_null(&normal_dot_tilde_e_cross_tilde_b),
              tilde_e_cross_tilde_b, normal_covector);
  for (size_t i = 0; i < number_of_grid_points; ++i) {
    get(*tilde_q)[i] = get(normal_dot_tilde_e_cross_tilde_b)[i] > 0.0
                           ? get(interior_tilde_q)[i]
                           : 0;
  }

  // spatial metric and sqrt determinant of spatial metric can be retrived
  // from Databox but only as gridless_tags with whole volume data (unlike
  // all the other arguments which are face tensors). Rather than doing
  // expensive tensor slicing operations on those, we just compute those two
  // quantities from inverse spatial metric as below.
  auto& interior_spatial_metric =
      get<gr::Tags::SpatialMetric<DataVector, 3>>(temp_buffer);
  auto& interior_sqrt_det_spatial_metric =
      get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(temp_buffer);

  determinant_and_inverse(make_not_null(&interior_sqrt_det_spatial_metric),
                          make_not_null(&interior_spatial_metric),
                          interior_inv_spatial_metric);
  get(interior_sqrt_det_spatial_metric) =
      1.0 / sqrt(get(interior_sqrt_det_spatial_metric));

  // Only compute the drift current, ignore the implicit current since that's
  // artificial.
  auto& exterior_tilde_j = get<Tags::TildeJ>(temp_buffer);
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

void Nonreflecting::fd_ghost(
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

  auto& normal_dot_tilde_e = get<::Tags::TempScalar<0>>(slice_buffer);
  auto& normal_dot_tilde_b = get<::Tags::TempScalar<1>>(slice_buffer);
  dot_product(make_not_null(&normal_dot_tilde_e), interior_tilde_e,
              subcell_normal_covector);
  dot_product(make_not_null(&normal_dot_tilde_b), interior_tilde_b,
              subcell_normal_covector);

  auto& normal_cross_tilde_b = get<::Tags::TempI<1, 3>>(slice_buffer);
  auto& tilde_e_projected = get<::Tags::TempI<2, 3>>(slice_buffer);
  auto& e_plus_vector = get<::Tags::TempI<3, 3>>(slice_buffer);

  // Polarized EM modes
  for (LeviCivitaIterator<3> it; it; ++it) {
    const auto& i = it[0];
    const auto& j = it[1];
    const auto& k = it[2];
    normal_cross_tilde_b.get(i) +=
        it.sign() * subcell_normal_vector.get(j) * interior_tilde_b.get(k);
  }
  for (size_t d = 0; d < 3; ++d) {
    tilde_e_projected.get(d) =
        interior_tilde_e.get(d) -
        get(normal_dot_tilde_e) * subcell_normal_vector.get(d);

    e_plus_vector.get(d) =
        0.5 * (tilde_e_projected.get(d) - normal_cross_tilde_b.get(d));

    exterior_tilde_e.get(d) = e_plus_vector.get(d);
    exterior_tilde_b.get(d) = 0.0;  // to prevent FPE or NaN in the next step
  }
  for (LeviCivitaIterator<3> it; it; ++it) {
    const auto& i = it[0];
    const auto& j = it[1];
    const auto& k = it[2];
    exterior_tilde_b.get(i) +=
        it.sign() * subcell_normal_vector.get(j) * e_plus_vector.get(k);
  }

  // Unphysical modes associated with div cleaning fields :
  // Just copy the normal component, ignore div cleaning fields.
  for (size_t d = 0; d < 3; ++d) {
    exterior_tilde_e.get(d) +=
        get(normal_dot_tilde_e) * subcell_normal_vector.get(d);
    exterior_tilde_b.get(d) +=
        get(normal_dot_tilde_b) * subcell_normal_vector.get(d);
  }
  get(*tilde_psi) = 0.0;
  get(*tilde_phi) = 0.0;

  // Charge density (drift)
  auto& tilde_e_cross_tilde_b = get<::Tags::TempI<4, 3>>(slice_buffer);
  for (LeviCivitaIterator<3> it; it; ++it) {
    const auto& i = it[0];
    const auto& j = it[1];
    const auto& k = it[2];
    tilde_e_cross_tilde_b.get(i) +=
        it.sign() * interior_tilde_e.get(j) * interior_tilde_b.get(k);
  }
  auto& normal_dot_tilde_e_cross_tilde_b =
      get<::Tags::TempScalar<2>>(slice_buffer);
  dot_product(make_not_null(&normal_dot_tilde_e_cross_tilde_b),
              tilde_e_cross_tilde_b, subcell_normal_covector);
  for (size_t i = 0; i < num_subcell_face_pts; ++i) {
    get(*tilde_q)[i] = get(normal_dot_tilde_e_cross_tilde_b)[i] > 0.0
                           ? get(interior_tilde_q)[i]
                           : 0;
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
