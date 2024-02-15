// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <vector>

#include "Evolution/Imex/GuessResult.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"

/// \cond
class DataVector;
template <typename>
class Variables;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace ForceFree {
namespace Imex {

struct InitialGuess {
  using return_tags = tmpl::list<Tags::TildeE>;
  using argument_tags = tmpl::list<Tags::TildeB, Tags::ParallelConductivity,
                                   gr::Tags::Lapse<DataVector>,
                                   gr::Tags::SpatialMetric<DataVector, 3>>;
  static std::vector<imex::GuessResult> apply(
      gsl::not_null<tnsr::I<DataVector, 3>*> tilde_e,
      const tnsr::I<DataVector, 3>& tilde_b, const double parallel_conductivity,
      const Scalar<DataVector>& lapse,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
      const Variables<tmpl::list<Tags::TildeE>>& inhomogeneous_terms,
      double implicit_weight);
};

}  // namespace Imex
}  // namespace ForceFree
