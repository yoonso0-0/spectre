// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/Limiters/Krivodonova.hpp"

#include <array>
#include <cstddef>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Krivodonova.hpp"
#include "Evolution/Systems/NewtonianEuler/Limiters/CharacteristicHelpers.hpp"
#include "Evolution/Systems/NewtonianEuler/Limiters/Flattener.hpp"
#include "Evolution/Systems/NewtonianEuler/Limiters/VariablesToLimit.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/CoefficientTransforms.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace NewtonianEuler::Limiters {

namespace detail {
template <size_t Dim, size_t ThermodynamicDim>
bool characteristic_krivodonova_impl(
    const gsl::not_null<Scalar<DataVector>*> mass_density_cons,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> momentum_density,
    const gsl::not_null<Scalar<DataVector>*> energy_density,
    const std::array<
        double, Spectral::maximum_number_of_points<Spectral::Basis::Legendre>>
        alphas,
    const Element<Dim>& element, const Mesh<Dim>& mesh,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
        equation_of_state,
    const std::unordered_map<
        std::pair<Direction<Dim>, ElementId<Dim>>,
        typename NewtonianEuler::Limiters::Krivodonova<Dim>::PackagedData,
        boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>& neighbor_data,
    const bool compute_char_transformation_numerically) {
  using CharacteristicVarsKrivodonova =
      ::Limiters::Krivodonova<Dim,
                              tmpl::list<NewtonianEuler::Tags::VMinus,
                                         NewtonianEuler::Tags::VMomentum<Dim>,
                                         NewtonianEuler::Tags::VPlus>>;
  // A limiter object with option alphas
  CharacteristicVarsKrivodonova char_vars_krivodonova{alphas};

  // Storage for transforming neighbor_data into char variables
  std::unordered_map<std::pair<Direction<Dim>, ElementId<Dim>>,
                     typename CharacteristicVarsKrivodonova::PackagedData,
                     boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>
      neighbor_char_coeffs_data{};
  for (const auto& [key, data] : neighbor_data) {
    neighbor_char_coeffs_data[key].modal_volume_data.initialize(
        mesh.number_of_grid_points());
    neighbor_char_coeffs_data[key].mesh = data.mesh;
  }

  Variables<tmpl::list<NewtonianEuler::Tags::MassDensityCons,
                       NewtonianEuler::Tags::MomentumDensity<Dim>,
                       NewtonianEuler::Tags::EnergyDensity>>
      cons_vars_buffer{mesh.number_of_grid_points()};
  Variables<tmpl::list<NewtonianEuler::Tags::VMinus,
                       NewtonianEuler::Tags::VMomentum<Dim>,
                       NewtonianEuler::Tags::VPlus>>
      char_vars_buffer{mesh.number_of_grid_points()};

  // Outer lambda: wraps applying Krivodonova to the NewtonianEuler
  // characteristics for one particular choice of characteristic decomposition
  const auto krivodonova_convert_neighbor_data_then_limit =
      [&char_vars_krivodonova, &char_vars_buffer, &cons_vars_buffer, &mesh,
       &element, &neighbor_data, &neighbor_char_coeffs_data](
          const gsl::not_null<Scalar<DataVector>*> char_v_minus,
          const gsl::not_null<tnsr::I<DataVector, Dim>*> char_v_momentum,
          const gsl::not_null<Scalar<DataVector>*> char_v_plus,
          const Matrix& left_eigenvectors) -> bool {
    // Convert neighbor data to characteristics
    for (const auto& [key, data] : neighbor_data) {
      // spectral coeffs of conserved vars -> nodal coeffs of conserved vars
      to_nodal_coefficients(
          make_not_null(&get(
              get<NewtonianEuler::Tags::MassDensityCons>(cons_vars_buffer))),
          get(get<::Tags::Modal<NewtonianEuler::Tags::MassDensityCons>>(
              data.modal_volume_data)),
          mesh);
      for (size_t i = 0; i < Dim; ++i) {
        to_nodal_coefficients(
            make_not_null(&get<NewtonianEuler::Tags::MomentumDensity<Dim>>(
                               cons_vars_buffer)
                               .get(i)),
            get<::Tags::Modal<NewtonianEuler::Tags::MomentumDensity<Dim>>>(
                data.modal_volume_data)
                .get(i),
            mesh);
      }
      to_nodal_coefficients(
          make_not_null(
              &get(get<NewtonianEuler::Tags::EnergyDensity>(cons_vars_buffer))),
          get(get<::Tags::Modal<NewtonianEuler::Tags::EnergyDensity>>(
              data.modal_volume_data)),
          mesh);

      // nodal coeffs of conserved vars -> nodal coeffs of char vars
      NewtonianEuler::Limiters::characteristic_fields(
          make_not_null(&char_vars_buffer), cons_vars_buffer,
          left_eigenvectors);

      // nodal coeffs of char vars -> modal coeffs of char vars
      to_modal_coefficients(
          make_not_null(&get(get<::Tags::Modal<NewtonianEuler::Tags::VMinus>>(
              neighbor_char_coeffs_data[key].modal_volume_data))),
          get(get<NewtonianEuler::Tags::VMinus>(char_vars_buffer)), mesh);
      for (size_t i = 0; i < Dim; ++i) {
        to_modal_coefficients(
            make_not_null(
                &get<::Tags::Modal<NewtonianEuler::Tags::VMomentum<Dim>>>(
                     neighbor_char_coeffs_data[key].modal_volume_data)
                     .get(i)),
            get<NewtonianEuler::Tags::VMomentum<Dim>>(char_vars_buffer).get(i),
            mesh);
      }
      to_modal_coefficients(
          make_not_null(&get(get<::Tags::Modal<NewtonianEuler::Tags::VPlus>>(
              neighbor_char_coeffs_data[key].modal_volume_data))),
          get(get<NewtonianEuler::Tags::VPlus>(char_vars_buffer)), mesh);
    }

    const bool result =
        char_vars_krivodonova(char_v_minus, char_v_momentum, char_v_plus,
                              element, mesh, neighbor_char_coeffs_data);
    return result;
  };

  return NewtonianEuler::Limiters::
      apply_limiter_to_characteristic_fields_in_all_directions(
          mass_density_cons, momentum_density, energy_density, mesh,
          equation_of_state, krivodonova_convert_neighbor_data_then_limit,
          compute_char_transformation_numerically);
}
}  // namespace detail

template <size_t Dim>
Krivodonova<Dim>::Krivodonova(
    const NewtonianEuler::Limiters::VariablesToLimit vars_to_limit,
    const std::array<
        double, Spectral::maximum_number_of_points<Spectral::Basis::Legendre>>
        alphas,
    const bool apply_flattener, const bool disable_for_debugging)
    : vars_to_limit_(vars_to_limit),
      alphas_(alphas),
      apply_flattener_(apply_flattener),
      disable_for_debugging_(disable_for_debugging),
      conservative_vars_krivodonova_(alphas_, disable_for_debugging_) {
  // ASSERT(1.0 >= alphas > 0.0) ?
}

template <size_t Dim>
void Krivodonova<Dim>::pup(PUP::er& p) {
  p | vars_to_limit_;
  p | alphas_;
  p | apply_flattener_;
  p | disable_for_debugging_;
  p | conservative_vars_krivodonova_;
}

template <size_t Dim>
void Krivodonova<Dim>::package_data(
    const gsl::not_null<PackagedData*> packaged_data,
    const Scalar<DataVector>& mass_density_cons,
    const tnsr::I<DataVector, Dim>& momentum_density,
    const Scalar<DataVector>& energy_density, const Mesh<Dim>& mesh,
    const OrientationMap<Dim>& orientation_map) const {
  conservative_vars_krivodonova_.package_data(packaged_data, mass_density_cons,
                                              momentum_density, energy_density,
                                              mesh, orientation_map);
}

template <size_t Dim>
template <size_t ThermodynamicDim>
bool Krivodonova<Dim>::operator()(
    const gsl::not_null<Scalar<DataVector>*> mass_density_cons,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> momentum_density,
    const gsl::not_null<Scalar<DataVector>*> energy_density,
    const Element<Dim>& element, const Mesh<Dim>& mesh,
    const Scalar<DataVector>& det_inv_logical_to_inertial_jacobian,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
        equation_of_state,
    const std::unordered_map<
        std::pair<Direction<Dim>, ElementId<Dim>>, PackagedData,
        boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>& neighbor_data)
    const {
  if (UNLIKELY(disable_for_debugging_)) {
    // Do not modify input tensors
    return false;
  }

  // Checks for the post-timestep, pre-limiter NewtonianEuler state:
#ifdef SPECTRE_DEBUG
  const double mean_density = mean_value(get(*mass_density_cons), mesh);
  ASSERT(mean_density > 0.0,
         "Positivity was violated on a cell-average level.");
  if constexpr (ThermodynamicDim == 2) {
    const double mean_energy = mean_value(get(*energy_density), mesh);
    ASSERT(mean_energy > 0.0,
           "Positivity was violated on a cell-average level.");
  }
#endif  // SPECTRE_DEBUG

  bool limiter_activated = false;

  if (vars_to_limit_ == NewtonianEuler::Limiters::VariablesToLimit::Conserved) {
    limiter_activated = conservative_vars_krivodonova_(
        mass_density_cons, momentum_density, energy_density, element, mesh,
        neighbor_data);
  } else if (vars_to_limit_ ==
                 NewtonianEuler::Limiters::VariablesToLimit::Characteristic or
             vars_to_limit_ == NewtonianEuler::Limiters::VariablesToLimit::
                                   NumericalCharacteristic) {
    const bool compute_char_transformation_numerically =
        (vars_to_limit_ ==
         NewtonianEuler::Limiters::VariablesToLimit::NumericalCharacteristic);
    limiter_activated = detail::characteristic_krivodonova_impl(
        mass_density_cons, momentum_density, energy_density, alphas_, element,
        mesh, equation_of_state, neighbor_data,
        compute_char_transformation_numerically);
  } else {
    ERROR(
        "No implementation of NewtonianEuler::Limiters::Krivodonova for "
        "variables: "
        << vars_to_limit_);
  }

  if (apply_flattener_) {
    const Scalar<DataVector> det_logical_to_inertial_jacobian{
        1.0 / get(det_inv_logical_to_inertial_jacobian)};
    const auto flattener_action = flatten_solution(
        mass_density_cons, momentum_density, energy_density, mesh,
        det_logical_to_inertial_jacobian, equation_of_state);
    if (flattener_action != FlattenerAction::NoOp) {
      limiter_activated = true;
    }
  }

  // Checks for the post-limiter NewtonianEuler state:
#ifdef SPECTRE_DEBUG
  ASSERT(min(get(*mass_density_cons)) > 0.0, "Bad density after limiting.");
  if constexpr (ThermodynamicDim == 2) {
    const auto specific_internal_energy = Scalar<DataVector>{
        get(*energy_density) / get(*mass_density_cons) -
        0.5 * get(dot_product(*momentum_density, *momentum_density)) /
            square(get(*mass_density_cons))};
    const auto pressure = equation_of_state.pressure_from_density_and_energy(
        *mass_density_cons, specific_internal_energy);
    ASSERT(min(get(pressure)) > 0.0, "Bad pressure after limiting.");
  }
#endif  // SPECTRE_DEBUG

  return limiter_activated;
}

template <size_t LocalDim>
bool operator==(const Krivodonova<LocalDim>& lhs,
                const Krivodonova<LocalDim>& rhs) {
  // No need to compare the conservative_vars_krivodonova_ member variable
  // because it is constructed from the other member variables.
  return lhs.vars_to_limit_ == rhs.vars_to_limit_ and
         lhs.alphas_ == rhs.alphas_ and
         lhs.apply_flattener_ == rhs.apply_flattener_ and
         lhs.disable_for_debugging_ == rhs.disable_for_debugging_;
}

template <size_t Dim>
bool operator!=(const Krivodonova<Dim>& lhs, const Krivodonova<Dim>& rhs) {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define THERMO_DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                               \
  template class Krivodonova<DIM(data)>;                   \
  template bool operator==(const Krivodonova<DIM(data)>&,  \
                           const Krivodonova<DIM(data)>&); \
  template bool operator!=(const Krivodonova<DIM(data)>&,  \
                           const Krivodonova<DIM(data)>&);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE

#define INSTANTIATE(_, data)                                                   \
  template bool Krivodonova<DIM(data)>::operator()(                            \
      const gsl::not_null<Scalar<DataVector>*> mass_density_cons,              \
      const gsl::not_null<tnsr::I<DataVector, DIM(data)>*> momentum_density,   \
      const gsl::not_null<Scalar<DataVector>*> energy_density,                 \
      const Element<DIM(data)>& element, const Mesh<DIM(data)>& mesh,          \
      const Scalar<DataVector>& det_inv_logical_to_inertial_jacobian,          \
      const EquationsOfState::EquationOfState<false, THERMO_DIM(data)>&        \
          equation_of_state,                                                   \
      const std::unordered_map<                                                \
          std::pair<Direction<DIM(data)>, ElementId<DIM(data)>>, PackagedData, \
          boost::hash<std::pair<Direction<DIM(data)>, ElementId<DIM(data)>>>>& \
          neighbor_data) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (1, 2))

#undef INSTANTIATE
#undef DIM
#undef THERMO_DIM

}  // namespace NewtonianEuler::Limiters
