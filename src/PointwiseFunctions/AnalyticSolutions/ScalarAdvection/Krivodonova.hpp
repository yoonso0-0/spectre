// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
class DataVector;
// IWYU pragma: no_forward_declare Tensor
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace ScalarAdvection {
namespace Solutions {
/*!
 * \brief Initial data for the 1D scalar advection problem adopted from
 * \cite Krivodonova2007 and its analytic solution. The initial proﬁle consists
 * of a combination of Gaussians, a square pulse, a sharp triangle, and a
 * combination of half-ellipses.
 *
 * \f{align*}
 * u(x,t=0) = \left\{\begin{array}{lcl}
 *  (G(x,\beta,z-\delta) + G(x,\beta,z+\delta) + 4G(x,\beta,z))/6 & \text{if} &
 *   -0.8 \leq x \leq -0.6 \\
 *  1 & \text{if} & -0.4 \leq x \leq -0.2 \\
 *  1 - |10(x-0.1)| & \text{if} & 0 \leq x \leq 0.2 \\
 *  (F(x,\alpha,a-\delta) + F(x,\alpha,a+\delta) + 4F(x,\alpha,a))/6 & \text{if}
 *  & 0.4 \leq x \leq 0.6 \\
 *  0 & \text{otherwise} & \\
 *  \end{array}\right\},
 * \f}
 *
 * where
 *
 * \f{align*}
 * G(x,\beta, z)  & = e^{-\beta(x-z)^2} \\
 * F(x,\alpha, a) & = \sqrt{\max(1-\alpha^2(x-a)^2, 0)}
 * \f}
 *
 * with \f$a=0.5, z=-0.7, \delta=0.005, \alpha=10, \text{and }\beta =
 * \log2/(36\delta^2)\f$.
 *
 */
class Krivodonova : public MarkAsAnalyticSolution {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{"An advecting Krivodonova 1D profile"};

  Krivodonova() = default;
  Krivodonova(const Krivodonova&) noexcept = default;
  Krivodonova& operator=(const Krivodonova&) noexcept = default;
  Krivodonova(Krivodonova&&) noexcept = default;
  Krivodonova& operator=(Krivodonova&&) noexcept = default;
  ~Krivodonova() noexcept = default;

  template <typename DataType>
  tuples::TaggedTuple<ScalarAdvection::Tags::U> variables(
      const tnsr::I<DataType, 1>& x, double t,
      tmpl::list<ScalarAdvection::Tags::U> /*meta*/) const noexcept;

  // clang-tidy: no pass by reference
  void pup(PUP::er& p) noexcept;  // NOLINT
};

bool operator==(const Krivodonova& /*lhs*/,
                const Krivodonova& /*rhs*/) noexcept;

bool operator!=(const Krivodonova& lhs, const Krivodonova& rhs) noexcept;

}  // namespace Solutions
}  // namespace ScalarAdvection
