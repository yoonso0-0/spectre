// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <pup.h>
#include <string>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/Systems/ForceFree/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace ForceFree::BoundaryConditions {
/*!
 * \brief ...
 *
 *
 */
class CopyFields final : public BoundaryCondition {
 private:
  using TildeE = ForceFree::Tags::TildeE;
  using TildeB = ForceFree::Tags::TildeB;
  using TildePsi = ForceFree::Tags::TildePsi;
  using TildePhi = ForceFree::Tags::TildePhi;
  using TildeQ = ForceFree::Tags::TildeQ;

  using SqrtDetSpatialMetric = gr::Tags::SqrtDetSpatialMetric<DataVector>;
  using SpatialMetric = gr::Tags::SpatialMetric<DataVector, 3>;
  using InvSpatialMetric = gr::Tags::InverseSpatialMetric<DataVector, 3>;
  using Lapse = gr::Tags::Lapse<DataVector>;
  using Shift = gr::Tags::Shift<DataVector, 3>;

 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{"CopyFields boundary conditions."};

  CopyFields() = default;
  CopyFields(CopyFields&&) = default;
  CopyFields& operator=(CopyFields&&) = default;
  CopyFields(const CopyFields&) = default;
  CopyFields& operator=(const CopyFields&) = default;
  ~CopyFields() override = default;

  explicit CopyFields(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, CopyFields);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override;

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::Ghost;

  void pup(PUP::er& p) override;

  using dg_interior_evolved_variables_tags = tmpl::list<TildeE, TildeB, TildeQ>;
  using dg_interior_temporary_tags = tmpl::list<Lapse, Shift, InvSpatialMetric>;
  using dg_gridless_tags = tmpl::list<>;

  static std::optional<std::string> dg_ghost(
      const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_e,
      const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_b,
      const gsl::not_null<Scalar<DataVector>*> tilde_psi,
      const gsl::not_null<Scalar<DataVector>*> tilde_phi,
      const gsl::not_null<Scalar<DataVector>*> tilde_q,

      const gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*>
          tilde_e_flux,
      const gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*>
          tilde_b_flux,
      const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          tilde_psi_flux,
      const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          tilde_phi_flux,
      const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          tilde_q_flux,

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
          interior_inv_spatial_metric);
};
}  // namespace ForceFree::BoundaryConditions
