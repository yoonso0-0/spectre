// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>

#include "DataStructures/DataBox/Tag.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags/OptionsGroup.hpp"
#include "Options/Options.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace ForceFree::subcell {

struct TciOptions {
  /*!
   * \brief The cutoff of the absolute value of the generalized charge density
   * \f$\tilde{Q}\f$ in an element to apply the Persson TCI.
   *
   * If maximum absolute value of \f$\tilde{Q}\f$ is below this option value,
   * the Persson TCI is not triggered for it.
   */
  struct CutoffTildeQ {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {
        "The cutoff of the absolute value of the TildeQ to use the Persson "
        "TCI."};
  };

  /*!
   * \brief The cutoff of the absolute value of J^i
   */
  struct CutoffTildeJ {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {
        "The cutoff of the magnitude of TildeJ to apply the Persson TCI ="};
  };

  /*!
   * \brief The cutoff of the Joule Heating
   */
  struct CutoffHeating {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {
        "The cutoff of J.E to apply the Persson TCI"};
  };

  using options = tmpl::list<CutoffTildeQ, CutoffTildeJ, CutoffHeating>;

  static constexpr Options::String help = {
      "Options for the troubled-cell indicator"};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/);

  double cutoff_tilde_q{std::numeric_limits<double>::signaling_NaN()};
  double cutoff_tilde_j{std::numeric_limits<double>::signaling_NaN()};
  double cutoff_heating{std::numeric_limits<double>::signaling_NaN()};
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
