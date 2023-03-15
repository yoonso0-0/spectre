// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/Imex/InitialGuess.hpp"

#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Imex/GuessResult.hpp"
#include "Utilities/TMPL.hpp"

namespace ForceFree::Imex {

imex::GuessResult InitialGuess::apply(
    const gsl::not_null<tnsr::I<DataVector, 3>*> tilde_e,
    const tnsr::I<DataVector, 3>& tilde_b, const double parallel_conductivity,
    const Scalar<DataVector>& lapse,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const Variables<tmpl::list<Tags::TildeE>>& inhomogeneous_terms,
    const double implicit_weight) {
  // std::cout << " ForceFree::InitialGuess called " << std::endl;
  Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
                       ::Tags::TempScalar<2>>>
      buffer{get(lapse).size()};
  auto& tilde_b_squared = get<::Tags::TempScalar<0>>(buffer);
  auto& tilde_e_dot_tilde_b = get<::Tags::TempScalar<1>>(buffer);
  dot_product(make_not_null(&tilde_b_squared), tilde_b, tilde_b,
              spatial_metric);
  dot_product(make_not_null(&tilde_e_dot_tilde_b), *tilde_e, tilde_b,
              spatial_metric);

  auto& temp_factor = get<::Tags::TempScalar<2>>(buffer);
  get(temp_factor) = implicit_weight * parallel_conductivity * get(lapse);

  for (size_t i = 0; i < 3; ++i) {
    (*tilde_e).get(i) = get<Tags::TildeE>(inhomogeneous_terms).get(i) -
                        get(temp_factor) * get(tilde_e_dot_tilde_b) *
                            tilde_b.get(i) /
                            ((1.0 + get(temp_factor)) * get(tilde_b_squared));
  }

  return imex::GuessResult::InitialGuess;
}

}  // namespace ForceFree::Imex
