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

// Implicit sector for parallel current density
struct ParallelCurrent : tt::ConformsTo<imex::protocols::ImplicitSector> {
  using tensors = tmpl::list<Tags::TildeE>;
  using tags_from_evolution =
      tmpl::list<Tags::TildeQ, Tags::TildeB, Tags::ParallelConductivity,
                 gr::Tags::Lapse<DataVector>,
                 gr::Tags::SqrtDetSpatialMetric<DataVector>,
                 gr::Tags::SpatialMetric<DataVector, 3>, Tags::NsInteriorMask>;
  using simple_tags = tmpl::list<>;
  using compute_tags = tmpl::list<>;

  using source_prep = tmpl::list<>;
  using jacobian_prep = tmpl::list<>;
  using initial_guess_prep = tmpl::list<>;

  using source = StiffSourceTildeE;

  // using ordinary IMEX solve
  using jacobian = StiffSourceTildeEJacobian;
  using initial_guess = InitialGuess;
  // using initial_guess = imex::GuessExplicitResult;

  using fallback = imex::NoFallback;
  // using fallback = ForceFree::Imex::IgnoreRectifierTerm;
};

struct IgnoreRectifierTermInParallelCurrent
    : tt::ConformsTo<imex::protocols::ImplicitSector> {
  using tensors = tmpl::list<Tags::TildeE>;
  using tags_from_evolution =
      tmpl::list<Tags::TildeQ, Tags::TildeB, Tags::ParallelConductivity,
                 gr::Tags::Lapse<DataVector>,
                 gr::Tags::SqrtDetSpatialMetric<DataVector>,
                 gr::Tags::SpatialMetric<DataVector, 3>, Tags::NsInteriorMask>;
  using simple_tags = tmpl::list<>;
  using compute_tags = tmpl::list<>;

  using source_prep = tmpl::list<>;
  using jacobian_prep = tmpl::list<>;
  using initial_guess_prep = tmpl::list<>;

  using source = StiffSourceTildeE;

  // using ordinary IMEX solve
  using jacobian = imex::NoJacobianBecauseSolutionIsAnalytic;
  using initial_guess = InitialGuess;

  using fallback = imex::NoFallback;
};

}  // namespace Imex
}  // namespace ForceFree
