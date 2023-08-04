// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "Evolution/Tags.hpp"
#include "Options/Options.hpp"

namespace imex {
namespace OptionTags {
struct ImexImplicitSolveTolerance {
  static constexpr Options::String help =
      "Absolute tolerance for IMEX solve (only used in Implicit ImexMode)";
  using type = double;
  using group = evolution::OptionTags::Group;
};
}  // namespace OptionTags

namespace Tags {
struct SolveTolerance : db::SimpleTag {
  using type = double;
  using option_tags =
      tmpl::list<::imex::OptionTags::ImexImplicitSolveTolerance>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& tolerance) { return tolerance; }
};
}  // namespace Tags
}  // namespace imex
