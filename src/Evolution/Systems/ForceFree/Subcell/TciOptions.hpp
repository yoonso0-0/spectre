// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <optional>

#include "DataStructures/DataBox/Tag.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags/OptionsGroup.hpp"
#include "Options/Auto.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace ForceFree::subcell {

struct TciOptions {
 private:
  struct DoNotCheckTildeQ {};
  struct DoNotCheckTildeE {};

 public:
  /*!
   * \brief The cutoff of the absolute value of the generalized charge density
   * \f$\tilde{Q}\f$ in an element to apply the Persson TCI.
   *
   * If maximum absolute value of \f$\tilde{Q}\f$ is below this option value,
   * the Persson TCI is not triggered for it.
   */
  struct TildeQCutoff {
    using type = Options::Auto<double, DoNotCheckTildeQ>;
    static constexpr Options::String help = {
        "If maximum absolute value of TildeQ in an element is below this value "
        "we do not apply the Persson TCI to TildeQ. To disable the check, set "
        "this option to 'DoNotCheckTildeQ'."};
  };

  struct AlphaMagE {
    using type = Options::Auto<double, DoNotCheckTildeE>;
    static constexpr Options::String help = {"If"};
  };

  struct AlphaMagB {
    using type = double;
    static constexpr Options::String help = {"If"};
  };

  struct DeltaAlpha {
    using type = double;
    static constexpr Options::String help = {"If"};
  };

  struct UseUmaxInsteadOfNorm {
    using type = bool;
    static constexpr Options::String help = {"If"};
  };

  using options = tmpl::list<TildeQCutoff, AlphaMagE, AlphaMagB, DeltaAlpha,
                             UseUmaxInsteadOfNorm>;

  static constexpr Options::String help = {
      "Options for the troubled-cell indicator"};

  TciOptions();
  explicit TciOptions(std::optional<double> tilde_q_cutoff_in);
  TciOptions(std::optional<double> tilde_q_cutoff_in,
             std::optional<double> alpha_mag_e_in, double alpha_mag_b_in,
             double delta_alpha_in, bool use_umax_instead_of_norm_in);

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/);

  std::optional<double> tilde_q_cutoff{
      std::numeric_limits<double>::signaling_NaN()};
  std::optional<double> alpha_mag_e{
      std::numeric_limits<double>::signaling_NaN()};
  double alpha_mag_b{std::numeric_limits<double>::signaling_NaN()};
  double delta_alpha{std::numeric_limits<double>::signaling_NaN()};
  bool use_umax_instead_of_norm{false};
};

namespace OptionTags {
struct TciOptions {
  using type = subcell::TciOptions;
  static constexpr Options::String help = "TCI options for ForceFree system";
  using group = ::dg::OptionTags::DiscontinuousGalerkinGroup;
};
}  // namespace OptionTags

namespace Tags {
struct TciOptions : db::SimpleTag {
  using type = subcell::TciOptions;
  using option_tags = tmpl::list<typename OptionTags::TciOptions>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& tci_options) {
    return tci_options;
  }
};
}  // namespace Tags
}  // namespace ForceFree::subcell
