// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/BoundaryConditions/CopyFields.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/ForceFree/Fluxes.hpp"

#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Variables.hpp"

namespace ForceFree::BoundaryConditions {

CopyFields::CopyFields(CkMigrateMessage* const msg) : BoundaryCondition(msg) {}

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
CopyFields::get_clone() const {
  return std::make_unique<CopyFields>(*this);
}

void CopyFields::pup(PUP::er& p) { BoundaryCondition::pup(p); }

// NOLINTNEXTLINE
PUP::able::PUP_ID CopyFields::my_PUP_ID = 0;

std::optional<std::string> CopyFields::dg_ghost(
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
    const tnsr::i<DataVector, 3, Frame::Inertial>& /*normal_covector*/,
    const tnsr::I<DataVector, 3, Frame::Inertial>& /*normal_vector*/,

    // interior evolved vars tags
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_tilde_e,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_tilde_b,
    const Scalar<DataVector>& interior_tilde_q,

    // interior temporary tags
    const Scalar<DataVector>& interior_lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_shift,
    const tnsr::II<DataVector, 3, Frame::Inertial>&
        interior_inv_spatial_metric) {
  get(*lapse) = get(interior_lapse);
  for (size_t i = 0; i < 3; ++i) {
    (*shift).get(i) = interior_shift.get(i);
    for (size_t j = 0; j < 3; ++j) {
      (*inv_spatial_metric).get(i, j) = interior_inv_spatial_metric.get(i, j);
    }
  }

  //
  // Copy variables
  //

  for (size_t i = 0; i < 3; ++i) {
    (*tilde_e).get(i) = interior_tilde_e.get(i);
    (*tilde_b).get(i) = interior_tilde_b.get(i);
  }
  get(*tilde_psi) = 0.0;
  get(*tilde_phi) = 0.0;
  get(*tilde_q) = get(interior_tilde_q);

  const size_t number_of_grid_points = get(interior_lapse).size();

  Variables<tmpl::list<Tags::TildeJ, gr::Tags::SpatialMetric<DataVector, 3>,
                       gr::Tags::SqrtDetSpatialMetric<DataVector>>>
      temp_buffer{number_of_grid_points};
  auto& exterior_tilde_j = get<Tags::TildeJ>(temp_buffer);
  auto& interior_spatial_metric =
      get<gr::Tags::SpatialMetric<DataVector, 3>>(temp_buffer);
  auto& interior_sqrt_det_spatial_metric =
      get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(temp_buffer);

  for (size_t i = 0; i < 3; ++i) {
    exterior_tilde_j.get(i) = 0.0;
  }

  // spatial metric and sqrt determinant of spatial metric can be retrived from
  // Databox but only as gridless_tags with whole volume data (unlike all the
  // other arguments which are face tensors). Rather than doing expensive tensor
  // slicing operations on those, we just compute those two quantities from
  // inverse spatial metric as below.
  determinant_and_inverse(make_not_null(&interior_sqrt_det_spatial_metric),
                          make_not_null(&interior_spatial_metric),
                          interior_inv_spatial_metric);
  get(interior_sqrt_det_spatial_metric) =
      1.0 / sqrt(get(interior_sqrt_det_spatial_metric));

  Fluxes::apply(tilde_e_flux, tilde_b_flux, tilde_psi_flux, tilde_phi_flux,
                tilde_q_flux, *tilde_e, *tilde_b, *tilde_psi, *tilde_phi,
                *tilde_q, exterior_tilde_j, interior_lapse, interior_shift,
                interior_sqrt_det_spatial_metric, interior_spatial_metric,
                interior_inv_spatial_metric);

  return {};
}

}  // namespace ForceFree::BoundaryConditions
