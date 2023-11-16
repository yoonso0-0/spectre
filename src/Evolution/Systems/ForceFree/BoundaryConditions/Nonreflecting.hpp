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

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/ForceFree/System.hpp"

/// \cond
class DataVector;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace ForceFree::BoundaryConditions {
/*!
 * \brief No-incoming characteristics boundary condition
 *
 *
 */
class Nonreflecting final : public BoundaryCondition {
 private:
  using TildeE = ForceFree::Tags::TildeE;
  using TildeB = ForceFree::Tags::TildeB;
  using TildePsi = ForceFree::Tags::TildePsi;
  using TildePhi = ForceFree::Tags::TildePhi;
  using TildeQ = ForceFree::Tags::TildeQ;
  using TildeJ = ForceFree::Tags::TildeJ;

  using SqrtDetSpatialMetric = gr::Tags::SqrtDetSpatialMetric<DataVector>;
  using SpatialMetric = gr::Tags::SpatialMetric<DataVector, 3>;
  using InvSpatialMetric = gr::Tags::InverseSpatialMetric<DataVector, 3>;
  using Lapse = gr::Tags::Lapse<DataVector>;
  using Shift = gr::Tags::Shift<DataVector, 3>;

  template <typename T>
  using Flux = ::Tags::Flux<T, tmpl::size_t<3>, Frame::Inertial>;

 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{"Nonreflecting boundary conditions."};

  Nonreflecting() = default;
  Nonreflecting(Nonreflecting&&) = default;
  Nonreflecting& operator=(Nonreflecting&&) = default;
  Nonreflecting(const Nonreflecting&) = default;
  Nonreflecting& operator=(const Nonreflecting&) = default;
  ~Nonreflecting() override = default;

  explicit Nonreflecting(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, Nonreflecting);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override;

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::Ghost;

  void pup(PUP::er& p) override;

  using dg_interior_evolved_variables_tags = tmpl::list<TildeE, TildeB, TildeQ>;
  using dg_interior_temporary_tags =
      tmpl::list<Lapse, Shift, InvSpatialMetric>;
  using dg_gridless_tags = tmpl::list<Tags::ParallelConductivity>;

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
          interior_inv_spatial_metric,
      const double parallel_conductivity);

  using fd_interior_evolved_variables_tags = tmpl::list<TildeE, TildeB, TildeQ>;
  using fd_interior_temporary_tags = tmpl::list<
      TildeJ, Lapse, Shift, InvSpatialMetric,
      domain::Tags::InverseJacobian<3, Frame::ElementLogical, Frame::Inertial>,
      domain::Tags::Mesh<3>, evolution::dg::subcell::Tags::Mesh<3>>;
  using fd_gridless_tags =
      tmpl::list<Tags::ParallelConductivity, fd::Tags::Reconstructor>;

  void fd_ghost(
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
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_tilde_e,
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_tilde_b,
      const Scalar<DataVector>& interior_tilde_q,

      // fd_interior_temporary_tags
      const tnsr::I<DataVector, 3, Frame::Inertial>& volume_tilde_j,
      const Scalar<DataVector>& volume_lapse,
      const tnsr::I<DataVector, 3, Frame::Inertial>& volume_shift,
      const tnsr::II<DataVector, 3, Frame::Inertial>&
          interior_inv_spatial_metric,
      const ::InverseJacobian<DataVector, 3, Frame::ElementLogical,
                              Frame::Inertial>& inv_jacobian_dg,
      const Mesh<3>& dg_mesh, const Mesh<3>& subcell_mesh,

      // fd_gridless_tags
      double parallel_conductivity,
      const fd::Reconstructor& reconstructor) const;
};
}  // namespace ForceFree::BoundaryConditions
