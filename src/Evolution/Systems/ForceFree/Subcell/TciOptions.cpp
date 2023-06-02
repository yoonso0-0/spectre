// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/Subcell/TciOptions.hpp"

#include <pup.h>

#include "Utilities/Serialization/PupStlCpp17.hpp"

namespace ForceFree::subcell {
void TciOptions::pup(PUP::er& p) { p | cutoff_tilde_q; }
}  // namespace ForceFree::subcell
