// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/ForceFree/Subcell/TciOptions.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.ForceFree.Subcell.TciOptions",
                  "[Unit][Evolution]") {
  const auto tci_options_from_opts =
      TestHelpers::test_option_tag<ForceFree::subcell::OptionTags::TciOptions>(
          "CutoffTildeQ: 1.0e-10\n");
  const auto tci_options = serialize_and_deserialize(tci_options_from_opts);
  CHECK(tci_options.cutoff_tilde_q == 1.0e-10);
}
