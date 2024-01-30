// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/ForceFree/Subcell/TciOptions.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/Serialization/Serialize.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ForceFree.Subcell.TciOptions",
                  "[Unit][Evolution]") {
  const auto tci_options_from_opts =
      TestHelpers::test_option_tag<ForceFree::subcell::OptionTags::TciOptions>(
          "TildeQCutoff: 1.0e-10\n"
          "AlphaMagE: 3.0\n"
          "AlphaMagB: 4.0\n"
          "DeltaAlpha: 1.0\n"
          "UseUmaxInsteadOfNorm: true\n");
  const auto tci_options = serialize_and_deserialize(tci_options_from_opts);
  CHECK(tci_options.tilde_q_cutoff == 1.0e-10);
  CHECK(tci_options.alpha_mag_e == 3.0);
  CHECK(tci_options.alpha_mag_b == 4.0);
  CHECK(tci_options.delta_alpha == 1.0);
  CHECK(tci_options.use_umax_instead_of_norm);
}
