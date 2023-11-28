// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataVector.hpp"
#include "Evolution/Imex/GuessResult.hpp"
#include "Evolution/Imex/Protocols/ImexSystem.hpp"
#include "Evolution/Systems/ForceFree/ElectricCurrentDensity.hpp"
#include "Evolution/Systems/ForceFree/Imex/InitialGuess.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace ForceFree {
namespace Imex {

struct ParallelCurrent : tt::ConformsTo<imex::protocols::ImplicitSector> {
  using tensors = tmpl::list<Tags::TildeE>;
  using initial_guess = ForceFree::Imex::InitialGuess;

  struct ExactSolve {
    using tags_from_evolution = tmpl::list<
        Tags::TildeQ, Tags::TildeB, Tags::ParallelConductivity,
        gr::Tags::Lapse<DataVector>, gr::Tags::SqrtDetSpatialMetric<DataVector>,
        gr::Tags::SpatialMetric<DataVector, 3>, Tags::NsInteriorMask>;

    using simple_tags = tmpl::list<>;
    using compute_tags = tmpl::list<>;

    using source_prep = tmpl::list<>;
    using jacobian_prep = tmpl::list<>;

    using source = StiffSourceTildeE;
    using jacobian = StiffSourceTildeEJacobian;
  };

  struct IgnoreRectifierTerm : ExactSolve {
    using jacobian = imex::NoJacobianBecauseSolutionIsAnalytic;
  };

  using solve_attempts = tmpl::list<ExactSolve, IgnoreRectifierTerm>;
};

}  // namespace Imex
}  // namespace ForceFree
