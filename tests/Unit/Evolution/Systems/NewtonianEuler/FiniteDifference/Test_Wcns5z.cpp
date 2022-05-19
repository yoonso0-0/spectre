// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "Evolution/Systems/NewtonianEuler/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/NewtonianEuler/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/NewtonianEuler/FiniteDifference/Wcns5z.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Evolution/Systems/NewtonianEuler/FiniteDifference/PrimReconstructor.hpp"

namespace {
template <size_t Dim>
void test() {
  namespace helpers = TestHelpers::NewtonianEuler::fd;
  const NewtonianEuler::fd::Wcns5zPrim<Dim> wcns5z_recons{2, 2.0e-16};
  helpers::test_prim_reconstructor<Dim>(5, wcns5z_recons);

  const auto wcns5z_from_options_base = TestHelpers::test_factory_creation<
      NewtonianEuler::fd::Reconstructor<Dim>,
      NewtonianEuler::fd::OptionTags::Reconstructor<Dim>>(
      "Wcns5zPrim:\n"
      "  NonlinearWeightExponent: 2\n"
      "  Epsilon: 2.0e-16\n");
  auto* const wcns5z_from_options =
      dynamic_cast<const NewtonianEuler::fd::Wcns5zPrim<Dim>*>(
          wcns5z_from_options_base.get());
  REQUIRE(wcns5z_from_options != nullptr);
  CHECK(*wcns5z_from_options == wcns5z_recons);

  CHECK(wcns5z_recons != NewtonianEuler::fd::Wcns5zPrim<Dim>(1, 2.0e-16));
  CHECK(wcns5z_recons != NewtonianEuler::fd::Wcns5zPrim<Dim>(2, 1.0e-16));
  CHECK(wcns5z_recons == NewtonianEuler::fd::Wcns5zPrim<Dim>(2, 2.0e-16));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianEuler.Fd.Wcns5zPrim",
                  "[Unit][Evolution]") {
  test<1>();
  test<2>();
  test<3>();
}
