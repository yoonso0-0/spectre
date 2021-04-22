// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

namespace ScalarAdvection {
namespace AnalyticData {
class Sinusoid;
}  // namespace AnalyticData

namespace Solutions {
class Sinusoid;
}  // namespace Solutions
}  // namespace ScalarAdvection

template <size_t Dim, typename InitialData>
struct EvolutionMetavars;
