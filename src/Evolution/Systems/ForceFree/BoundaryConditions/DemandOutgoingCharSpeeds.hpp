// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <optional>
#include <pup.h>
#include <string>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/ForceFree/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/ForceFree/System.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
/// \endcond

namespace ForceFree::BoundaryConditions {
/*!
 * \brief A boundary condition that only verifies that all characteristic speeds
 * are directed out of the domain; no boundary data is altered by this boundary
 * condition.
 */
class DemandOutgoingCharSpeeds final : public BoundaryCondition {
 private:
  using TildeE = ForceFree::Tags::TildeE;
  using TildeB = ForceFree::Tags::TildeB;
  using TildePsi = ForceFree::Tags::TildePsi;
  using TildePhi = ForceFree::Tags::TildePhi;
  using TildeQ = ForceFree::Tags::TildeQ;
  using TildeJ = ForceFree::Tags::TildeJ;

  using Lapse = gr::Tags::Lapse<DataVector>;
  using Shift = gr::Tags::Shift<DataVector, 3>;
  using SqrtDetSpatialMetric = gr::Tags::SqrtDetSpatialMetric<DataVector>;
  using SpatialMetric = gr::Tags::SpatialMetric<DataVector, 3>;
  using InvSpatialMetric = gr::Tags::InverseSpatialMetric<DataVector, 3>;

  template <typename T>
  using Flux = ::Tags::Flux<T, tmpl::size_t<3>, Frame::Inertial>;

 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "DemandOutgoingCharSpeeds boundary condition that only verifies the "
      "characteristic speeds are all directed out of the domain."};

  DemandOutgoingCharSpeeds() = default;
  DemandOutgoingCharSpeeds(DemandOutgoingCharSpeeds&&) = default;
  DemandOutgoingCharSpeeds& operator=(DemandOutgoingCharSpeeds&&) = default;
  DemandOutgoingCharSpeeds(const DemandOutgoingCharSpeeds&) = default;
  DemandOutgoingCharSpeeds& operator=(const DemandOutgoingCharSpeeds&) =
      default;
  ~DemandOutgoingCharSpeeds() override = default;

  explicit DemandOutgoingCharSpeeds(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, DemandOutgoingCharSpeeds);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override;

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::DemandOutgoingCharSpeeds;

  void pup(PUP::er& p) override;

  using dg_interior_evolved_variables_tags = tmpl::list<>;
  using dg_interior_temporary_tags = tmpl::list<Shift, Lapse>;
  using dg_interior_primitive_variables_tags = tmpl::list<>;
  using dg_gridless_tags = tmpl::list<>;

  static std::optional<std::string> dg_demand_outgoing_char_speeds(
      const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
          face_mesh_velocity,
      const tnsr::i<DataVector, 3, Frame::Inertial>&
          outward_directed_normal_covector,
      const tnsr::I<DataVector, 3, Frame::Inertial>&
      /*outward_directed_normal_vector*/,

      const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
      const Scalar<DataVector>& lapse);

  using fd_interior_evolved_variables_tags =
      tmpl::list<TildeE, TildeB, TildePsi, TildePhi, TildeQ>;
  using fd_interior_temporary_tags = tmpl::list<
      Lapse, Shift, InvSpatialMetric,
      domain::Tags::InverseJacobian<3, Frame::ElementLogical, Frame::Inertial>,
      domain::Tags::Mesh<3>, evolution::dg::subcell::Tags::Mesh<3>>;
  using fd_gridless_tags =
      tmpl::list<Tags::ParallelConductivity, fd::Tags::Reconstructor>;

  static void fd_demand_outgoing_char_speeds(
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
      const tnsr::II<DataVector, 3, Frame::Inertial>&
          interior_inv_spatial_metric,
      const ::InverseJacobian<DataVector, 3, Frame::ElementLogical,
                              Frame::Inertial>& inv_jacobian_dg,
      const Mesh<3>& dg_mesh, const Mesh<3>& subcell_mesh,

      // fd_gridless_tags
      double parallel_conductivity, const fd::Reconstructor& reconstructor);
};
}  // namespace ForceFree::BoundaryConditions
