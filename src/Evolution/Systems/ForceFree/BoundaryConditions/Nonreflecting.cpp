// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/BoundaryConditions/Nonreflecting.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>

#include "DataStructures/Tags/TempTensor.hpp"

#include "DataStructures/DataVector.hpp"
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

  // Unphysical modes associated with div cleaning fields : copy the normal
  // component.
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

}  // namespace ForceFree::BoundaryConditions
