// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/Imex/InitialGuess.hpp"

#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Imex/GuessResult.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

namespace ForceFree::Imex {

std::vector<imex::GuessResult> InitialGuess::apply(
    const gsl::not_null<tnsr::I<DataVector, 3>*> tilde_e,
    const tnsr::I<DataVector, 3>& tilde_b, const double parallel_conductivity,
    const Scalar<DataVector>& lapse,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const Variables<tmpl::list<Tags::TildeE>>& inhomogeneous_terms,
    const double implicit_weight) {
  const size_t num_grid_pts = get(lapse).size();

  Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
                       ::Tags::TempScalar<2>, ::Tags::TempScalar<3>>>
      buffer{num_grid_pts};

  auto& tilde_b_squared = get<::Tags::TempScalar<0>>(buffer);
  auto& tilde_e_dot_tilde_b = get<::Tags::TempScalar<1>>(buffer);
  dot_product(make_not_null(&tilde_b_squared), tilde_b, tilde_b,
              spatial_metric);
  dot_product(make_not_null(&tilde_e_dot_tilde_b),
              get<Tags::TildeE>(inhomogeneous_terms), tilde_b, spatial_metric);

  auto& temp_factor = get<::Tags::TempScalar<2>>(buffer);
  get(temp_factor) = implicit_weight * parallel_conductivity * get(lapse);

  for (size_t i = 0; i < 3; ++i) {
    (*tilde_e).get(i) = get<Tags::TildeE>(inhomogeneous_terms).get(i) -
                        get(temp_factor) * get(tilde_e_dot_tilde_b) *
                            tilde_b.get(i) /
                            ((1.0 + get(temp_factor)) * get(tilde_b_squared));
  }

  auto& tilde_e_squared = get<::Tags::TempScalar<3>>(buffer);
  dot_product(make_not_null(&tilde_e_squared),
              get<Tags::TildeE>(inhomogeneous_terms),
              get<Tags::TildeE>(inhomogeneous_terms), spatial_metric);

  // ASSERT(get(lapse).size() == 1,
  //        "ForceFree::Imex::Initialguess assumes that the size of input
  //        tensors " "is 1, but the size is "
  //            << get(lapse).size());

  std::vector<imex::GuessResult> result{num_grid_pts,
                                        imex::GuessResult::ExactSolution};

  for (size_t i = 0; i < num_grid_pts; ++i) {
    if (get(tilde_e_squared)[i] > get(tilde_b_squared)[i]) {
      result.at(i) = imex::GuessResult::InitialGuess;
    }
  }

  return result;
}

}  // namespace ForceFree::Imex
