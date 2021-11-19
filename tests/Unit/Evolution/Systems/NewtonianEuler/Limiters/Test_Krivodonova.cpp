// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "Evolution/Systems/NewtonianEuler/Limiters/Krivodonova.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"

namespace {
template <size_t Dim>
void test() {
  const auto krivodonova =
      TestHelpers::test_creation<NewtonianEuler::Limiters::Krivodonova<Dim>>(
          "VariablesToLimit: Characteristic\n"
          "Alphas: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]\n"
          "ApplyFlattener: True\n"
          "DisableForDebugging: False");
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianEuler.Limiters.Krivodonova",
                  "[Unit][Evolution]") {
  test<1>();
  test<2>();
  test<3>();
}
