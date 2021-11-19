// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Krivodonova.hpp"
#include "Evolution/Systems/NewtonianEuler/Limiters/VariablesToLimit.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
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
template <size_t Dim>
class Mesh;
template <size_t Dim>
class OrientationMap;
namespace EquationsOfState {
template <bool IsRelativistic, size_t ThermodynamicDim>
class EquationOfState;
}  // namespace EquationsOfState
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace NewtonianEuler {
namespace Limiters {
/*!
 * \brief Krivodonova limiter for the NewtonianEuler system.
 */
template <size_t Dim>
class Krivodonova {
 public:
  using ConservativeVarsKrivodonova = ::Limiters::Krivodonova<
      Dim, tmpl::list<NewtonianEuler::Tags::MassDensityCons,
                      NewtonianEuler::Tags::MomentumDensity<Dim>,
                      NewtonianEuler::Tags::EnergyDensity>>;

  struct VariablesToLimit {
    using type = NewtonianEuler::Limiters::VariablesToLimit;
    static type suggested_value() {
      return NewtonianEuler::Limiters::VariablesToLimit::Characteristic;
    }
    static constexpr Options::String help = {
        "Variable representation on which to apply the limiter"};
  };

  struct ApplyFlattener {
    using type = bool;
    static constexpr Options::String help = {
        "Flatten after limiting to restore pointwise positivity"};
  };

  using options =
      tmpl::list<VariablesToLimit, typename ConservativeVarsKrivodonova::Alphas,
                 ApplyFlattener,
                 typename ConservativeVarsKrivodonova::DisableForDebugging>;

  static constexpr Options::String help = {
      "A Krivodonova limiter specialized to the NewtonianEuler system"};
  static std::string name() { return "NewtonianEulerKrivodonova"; };

  explicit Krivodonova(
      NewtonianEuler::Limiters::VariablesToLimit vars_to_limit,
      std::array<double,
                 Spectral::maximum_number_of_points<Spectral::Basis::Legendre>>
          alphas,
      bool apply_flattener, bool disable_for_debugging = false);

  Krivodonova() = default;
  Krivodonova(const Krivodonova& /*rhs*/) = default;
  Krivodonova& operator=(const Krivodonova& /*rhs*/) = default;
  Krivodonova(Krivodonova&& /*rhs*/) = default;
  Krivodonova& operator=(Krivodonova&& /*rhs*/) = default;
  ~Krivodonova() = default;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  using PackagedData = typename ConservativeVarsKrivodonova::PackagedData;
  using package_argument_tags =
      typename ConservativeVarsKrivodonova::package_argument_tags;

  /// \brief Package data for sending to neighbor elements.
  void package_data(gsl::not_null<PackagedData*> packaged_data,
                    const Scalar<DataVector>& mass_density_cons,
                    const tnsr::I<DataVector, Dim>& momentum_density,
                    const Scalar<DataVector>& energy_density,
                    const Mesh<Dim>& mesh,
                    const OrientationMap<Dim>& orientation_map) const;

  using limit_tags = tmpl::list<NewtonianEuler::Tags::MassDensityCons,
                                NewtonianEuler::Tags::MomentumDensity<Dim>,
                                NewtonianEuler::Tags::EnergyDensity>;
  using limit_argument_tags = tmpl::list<
      domain::Tags::Element<Dim>, domain::Tags::Mesh<Dim>,
      domain::Tags::DetInvJacobian<Frame::ElementLogical, Frame::Inertial>,
      ::hydro::Tags::EquationOfStateBase>;

  template <size_t ThermodynamicDim>
  bool operator()(
      gsl::not_null<Scalar<DataVector>*> mass_density_cons,
      gsl::not_null<tnsr::I<DataVector, Dim>*> momentum_density,
      gsl::not_null<Scalar<DataVector>*> energy_density,
      const Element<Dim>& element, const Mesh<Dim>& mesh,
      const Scalar<DataVector>& det_inv_logical_to_inertial_jacobian,
      const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
          equation_of_state,
      const std::unordered_map<
          std::pair<Direction<Dim>, ElementId<Dim>>, PackagedData,
          boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>&
          neighbor_data) const;

 private:
  template <size_t LocalDim>
  // NOLINTNEXTLINE(readability-redundant-declaration) false positive
  friend bool operator==(const Krivodonova<LocalDim>& lhs,
                         const Krivodonova<LocalDim>& rhs);

  NewtonianEuler::Limiters::VariablesToLimit vars_to_limit_;
  std::array<double,
             Spectral::maximum_number_of_points<Spectral::Basis::Legendre>>
      alphas_;
  bool apply_flattener_;
  bool disable_for_debugging_;
  ConservativeVarsKrivodonova conservative_vars_krivodonova_;
};

template <size_t Dim>
bool operator!=(const Krivodonova<Dim>& lhs, const Krivodonova<Dim>& rhs);

}  // namespace Limiters
}  // namespace NewtonianEuler
