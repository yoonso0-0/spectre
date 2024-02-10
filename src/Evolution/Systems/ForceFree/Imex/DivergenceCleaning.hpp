// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "Evolution/Imex/GuessResult.hpp"
#include "Evolution/Imex/Protocols/ImexSystem.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace ForceFree {
namespace Imex {

struct DivCleaningStiffSources {
  using return_tags = tmpl::list<::Tags::Source<Tags::TildePsi>,
                                 ::Tags::Source<Tags::TildePhi>>;
  using argument_tags =
      tmpl::list<Tags::TildePsi, Tags::KappaPsi, Tags::TildePhi, Tags::KappaPhi,
                 gr::Tags::Lapse<DataVector>>;
  static void apply(gsl::not_null<Scalar<DataVector>*> stiff_source_tilde_psi,
                    gsl::not_null<Scalar<DataVector>*> stiff_source_tilde_phi,
                    const Scalar<DataVector>& tilde_psi, const double kappa_psi,
                    const Scalar<DataVector>& tilde_phi, const double kappa_phi,
                    const Scalar<DataVector>& lapse);
};

struct DivCleaningExactSolution {
  using return_tags = tmpl::list<Tags::TildePsi, Tags::TildePhi>;
  using argument_tags =
      tmpl::list<Tags::KappaPsi, Tags::KappaPhi, gr::Tags::Lapse<DataVector>>;

  static std::vector<imex::GuessResult> apply(
      gsl::not_null<Scalar<DataVector>*> tilde_psi,
      gsl::not_null<Scalar<DataVector>*> tilde_phi, const double kappa_psi,
      const double kappa_phi, const Scalar<DataVector>& lapse,
      const Variables<tmpl::list<Tags::TildePsi, Tags::TildePhi>>&
          inhomogeneous_terms,
      double implicit_weight);
};

struct DivCleaning : tt::ConformsTo<imex::protocols::ImplicitSector> {
  using tensors = tmpl::list<Tags::TildePsi, Tags::TildePhi>;
  using initial_guess = DivCleaningExactSolution;

  struct ExactSolve {
    using tags_from_evolution =
        tmpl::list<Tags::KappaPsi, Tags::KappaPhi, gr::Tags::Lapse<DataVector>>;

    using simple_tags = tmpl::list<>;
    using compute_tags = tmpl::list<>;

    using source_prep = tmpl::list<>;
    using jacobian_prep = tmpl::list<>;

    using source = DivCleaningStiffSources;
    using jacobian = imex::NoJacobianBecauseSolutionIsAnalytic;
  };

  using solve_attempts = tmpl::list<ExactSolve>;
};

}  // namespace Imex
}  // namespace ForceFree
