// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <pup.h>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/NeighborData.hpp"
#include "Evolution/Systems/NewtonianEuler/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "NumericalAlgorithms/FiniteDifference/FallbackReconstructorType.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Direction;
template <size_t Dim>
class Element;
template <size_t Dim>
class ElementId;
namespace EquationsOfState {
template <bool IsRelativistic, size_t ThermodynamicDim>
class EquationOfState;
}  // namespace EquationsOfState
template <size_t Dim>
class Mesh;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
template <typename TagsList>
class Variables;
/// \endcond

namespace NewtonianEuler::fd {
/*!
 * \brief Fifth order weighted nonlinear compact scheme reconstruction using the
 * Z oscillation indicator. See ::fd::reconstruction::wcns5z() for details.
 *
 */
template <size_t Dim>
class Wcns5zPrim : public Reconstructor<Dim> {
 private:
  // Conservative vars tags
  using MassDensityCons = NewtonianEuler::Tags::MassDensityCons;
  using EnergyDensity = NewtonianEuler::Tags::EnergyDensity;
  using MomentumDensity = NewtonianEuler::Tags::MomentumDensity<Dim>;

  // Primitive vars tags
  using MassDensity = NewtonianEuler::Tags::MassDensity<DataVector>;
  using Velocity = NewtonianEuler::Tags::Velocity<DataVector, Dim>;
  using SpecificInternalEnergy =
      NewtonianEuler::Tags::SpecificInternalEnergy<DataVector>;
  using Pressure = NewtonianEuler::Tags::Pressure<DataVector>;

  using prims_tags =
      tmpl::list<MassDensity, Velocity, SpecificInternalEnergy, Pressure>;
  using cons_tags = tmpl::list<MassDensityCons, MomentumDensity, EnergyDensity>;
  using flux_tags = db::wrap_tags_in<::Tags::Flux, cons_tags, tmpl::size_t<Dim>,
                                     Frame::Inertial>;
  using prim_tags_for_reconstruction =
      tmpl::list<MassDensity, Velocity, Pressure>;

 public:
  struct NonlinearWeightExponent {
    using type = size_t;
    static constexpr Options::String help = {
        "The exponent q to which the oscillation indicator term is raised"};
  };
  struct Epsilon {
    using type = double;
    static constexpr Options::String help = {
        "The parameter added to the oscillation indicators to avoid division "
        "by zero"};
  };
  struct LowOrderReconstructionToFallback {
    using type = ::fd::reconstruction::FallbackReconstructorType;
    static constexpr Options::String help = {
        "A low-order FD reconstruction scheme to fallback adaptively. Finite "
        "difference will switch to this reconstruction scheme if there are "
        "more extrema in a FD stencil than a specified number. See also the "
        "option 'MaxNumberOfExtrema' below. Adaptive fallback is disabled if "
        "'None'."};
  };
  struct MaxNumberOfExtrema {
    using type = size_t;
    static constexpr Options::String help = {
        "The maximum allowed number of extrema in FD stencil for using Wcns5z "
        "reconstruction before switching to a low-order reconstruction. If "
        "LowOrderReconstructionToFallback=None, this option is ignored"};
  };

  using options =
      tmpl::list<NonlinearWeightExponent, Epsilon,
                 LowOrderReconstructionToFallback, MaxNumberOfExtrema>;

  static constexpr Options::String help{
      "WCNS 5Z reconstruction scheme using primitive variables."};

  Wcns5zPrim() = default;
  Wcns5zPrim(Wcns5zPrim&&) = default;
  Wcns5zPrim& operator=(Wcns5zPrim&&) = default;
  Wcns5zPrim(const Wcns5zPrim&) = default;
  Wcns5zPrim& operator=(const Wcns5zPrim&) = default;
  ~Wcns5zPrim() override = default;

  Wcns5zPrim(size_t nonlinear_weight_exponent, double epsilon,
             ::fd::reconstruction::FallbackReconstructorType
                 low_order_recons_to_fallback,
             size_t max_number_of_extrema);

  explicit Wcns5zPrim(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(Reconstructor<Dim>, Wcns5zPrim);

  auto get_clone() const -> std::unique_ptr<Reconstructor<Dim>> override;

  void pup(PUP::er& p) override;

  size_t ghost_zone_size() const override { return 3; }

  using reconstruction_argument_tags = tmpl::list<
      ::Tags::Variables<prims_tags>, hydro::Tags::EquationOfStateBase,
      domain::Tags::Element<Dim>,
      evolution::dg::subcell::Tags::NeighborDataForReconstruction<Dim>,
      evolution::dg::subcell::Tags::Mesh<Dim>>;

  template <size_t ThermodynamicDim, typename TagsList>
  void reconstruct(
      gsl::not_null<std::array<Variables<TagsList>, Dim>*> vars_on_lower_face,
      gsl::not_null<std::array<Variables<TagsList>, Dim>*> vars_on_upper_face,
      const Variables<prims_tags>& volume_prims,
      const EquationsOfState::EquationOfState<false, ThermodynamicDim>& eos,
      const Element<Dim>& element,
      const FixedHashMap<
          maximum_number_of_neighbors(Dim),
          std::pair<Direction<Dim>, ElementId<Dim>>, std::vector<double>,
          boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>&
          neighbor_data,
      const Mesh<Dim>& subcell_mesh) const;

  template <size_t ThermodynamicDim, typename TagsList>
  void reconstruct_fd_neighbor(
      gsl::not_null<Variables<TagsList>*> vars_on_face,
      const Variables<prims_tags>& subcell_volume_prims,
      const EquationsOfState::EquationOfState<false, ThermodynamicDim>& eos,
      const Element<Dim>& element,
      const FixedHashMap<
          maximum_number_of_neighbors(Dim),
          std::pair<Direction<Dim>, ElementId<Dim>>, std::vector<double>,
          boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>&
          neighbor_data,
      const Mesh<Dim>& subcell_mesh,
      const Direction<Dim> direction_to_reconstruct) const;

 private:
  template <size_t LocalDim>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const Wcns5zPrim<LocalDim>& lhs,
                         const Wcns5zPrim<LocalDim>& rhs);

  size_t nonlinear_weight_exponent_ = 0;
  double epsilon_ = std::numeric_limits<double>::signaling_NaN();
  ::fd::reconstruction::FallbackReconstructorType low_order_recons_to_fallback_;
  size_t max_number_of_extrema_ = 0;

  void (*reconstruct_)(gsl::not_null<std::array<gsl::span<double>, Dim>*>,
                       gsl::not_null<std::array<gsl::span<double>, Dim>*>,
                       const gsl::span<const double>&,
                       const DirectionMap<Dim, gsl::span<const double>>&,
                       const Index<Dim>&, size_t, double, size_t) = nullptr;
  void (*reconstruct_lower_neighbor_)(gsl::not_null<DataVector*>,
                                      const DataVector&, const DataVector&,
                                      const Index<Dim>&, const Index<Dim>&,
                                      const Direction<Dim>&, const double&,
                                      const size_t&) = nullptr;
  void (*reconstruct_upper_neighbor_)(gsl::not_null<DataVector*>,
                                      const DataVector&, const DataVector&,
                                      const Index<Dim>&, const Index<Dim>&,
                                      const Direction<Dim>&, const double&,
                                      const size_t&) = nullptr;
};

template <size_t Dim>
bool operator!=(const Wcns5zPrim<Dim>& lhs, const Wcns5zPrim<Dim>& rhs) {
  return not(lhs == rhs);
}

}  // namespace NewtonianEuler::fd
