// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Utilities/TypeTraits/CreateGetStaticMemberVariableOrDefault.hpp"

namespace imex {

CREATE_GET_STATIC_MEMBER_VARIABLE_OR_DEFAULT(imex_time_stepping)

template <typename Metavars>
constexpr bool using_imex_v =
    get_imex_time_stepping_or_default_v<Metavars, false>;

}  // namespace imex
