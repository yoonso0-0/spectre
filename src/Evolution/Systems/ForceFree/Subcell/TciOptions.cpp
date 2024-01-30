// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/Subcell/TciOptions.hpp"

#include <pup.h>

#include "Utilities/Serialization/PupStlCpp17.hpp"

namespace ForceFree::subcell {

TciOptions::TciOptions() = default;

TciOptions::TciOptions(std::optional<double> tilde_q_cutoff_in)
    : tilde_q_cutoff(std::move(tilde_q_cutoff_in)) {}

TciOptions::TciOptions(std::optional<double> tilde_q_cutoff_in,
                       std::optional<double> alpha_mag_e_in,
                       const double alpha_mag_b_in, const double delta_alpha_in,
                       const bool use_umax_instead_of_norm_in)
    : tilde_q_cutoff(std::move(tilde_q_cutoff_in)),
      alpha_mag_e(std::move(alpha_mag_e_in)),
      alpha_mag_b(alpha_mag_b_in),
      delta_alpha(delta_alpha_in),
      use_umax_instead_of_norm(use_umax_instead_of_norm_in) {}

void TciOptions::pup(PUP::er& p) {
  p | tilde_q_cutoff;
  p | alpha_mag_b;
  p | alpha_mag_e;
  p | delta_alpha;
  p | use_umax_instead_of_norm;
}

}  // namespace ForceFree::subcell
