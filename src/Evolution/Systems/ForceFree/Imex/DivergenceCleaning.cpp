// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/Imex/DivergenceCleaning.hpp"

#include "Evolution/Imex/SolveImplicitSector.hpp"
#include "Evolution/Imex/SolveImplicitSector.tpp"
#include "Evolution/Systems/ForceFree/System.hpp"

namespace ForceFree::Imex {

void DivCleaningStiffSources::apply(
    const gsl::not_null<Scalar<DataVector>*> stiff_source_tilde_psi,
    const gsl::not_null<Scalar<DataVector>*> stiff_source_tilde_phi,
    const Scalar<DataVector>& tilde_psi, const double kappa_psi,
    const Scalar<DataVector>& tilde_phi, const double kappa_phi,
    const Scalar<DataVector>& lapse) {
  get(*stiff_source_tilde_psi) = -kappa_psi * get(lapse) * get(tilde_psi);
  get(*stiff_source_tilde_phi) = -kappa_phi * get(lapse) * get(tilde_phi);
}

std::vector<imex::GuessResult> DivCleaningExactSolution::apply(
    const gsl::not_null<Scalar<DataVector>*> tilde_psi,
    const gsl::not_null<Scalar<DataVector>*> tilde_phi, const double kappa_psi,
    const double kappa_phi, const Scalar<DataVector>& lapse,
    const Variables<tmpl::list<Tags::TildePsi, Tags::TildePhi>>&
        inhomogeneous_terms,
    double implicit_weight) {
  const size_t num_grid_pts = get(lapse).size();
  Variables<tmpl::list<::Tags::TempScalar<0>>> buffer{num_grid_pts};
  auto& temp_factor = get<::Tags::TempScalar<0>>(buffer);

  get(temp_factor) = implicit_weight * get(lapse);
  get(*tilde_psi) = get(get<Tags::TildePsi>(inhomogeneous_terms)) /
                    (1.0 + kappa_psi * get(temp_factor));
  get(*tilde_phi) = get(get<Tags::TildePhi>(inhomogeneous_terms)) /
                    (1.0 + kappa_phi * get(temp_factor));

  return std::vector<imex::GuessResult>{num_grid_pts,
                                        imex::GuessResult::ExactSolution};
}

}  // namespace ForceFree::Imex

template struct imex::SolveImplicitSector<ForceFree::System::variables_tag,
                                          ForceFree::Imex::DivCleaning>;
