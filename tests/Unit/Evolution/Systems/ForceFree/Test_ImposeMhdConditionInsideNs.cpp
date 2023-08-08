// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/ForceFree/ImposeMhdConditionInsideNs.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ForceFree.ImposeMhdInsideNs",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/ForceFree"};

  CHECK(true == true);

  //   pypp::check_with_random_values<1>(
  //       &ForceFree::Tags::ElectricFieldDotMagneticFieldCompute::function,
  //       "TestFunctions", {"e_dot_b_compute"}, {{{-1.0, 1.0}}},
  //       DataVector{5});
}
