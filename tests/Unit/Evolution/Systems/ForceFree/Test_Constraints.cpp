// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/ForceFree/Constraints.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ForceFree.Constraints",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/ForceFree"};

  pypp::check_with_random_values<1>(
      ForceFree::Tags::TildeESquaredCompute::function,
      "ElectromagneticVariables", {"tilde_e_or_tilde_b_squared"},
      {{{-1.0, 1.0}}}, DataVector{5});

  pypp::check_with_random_values<1>(
      ForceFree::Tags::TildeBSquaredCompute::function,
      "ElectromagneticVariables", {"tilde_e_or_tilde_b_squared"},
      {{{-1.0, 1.0}}}, DataVector{5});

  pypp::check_with_random_values<1>(
      ForceFree::Tags::TildeEDotTildeBCompute::function,
      "ElectromagneticVariables", {"tilde_e_dot_tilde_b_compute"},
      {{{-1.0, 1.0}}}, DataVector{5});

  pypp::check_with_random_values<1>(
      ForceFree::Tags::ElectricFieldDotMagneticFieldCompute::function,
      "ElectromagneticVariables", {"e_dot_b_compute"}, {{{-1.0, 1.0}}},
      DataVector{5});

  pypp::check_with_random_values<1>(
      ForceFree::Tags::MagneticDominanceViolationCompute::function,
      "ElectromagneticVariables", {"magnetic_dominance_violation_compute"},
      {{{-1.0, 1.0}}}, DataVector{5});
}
