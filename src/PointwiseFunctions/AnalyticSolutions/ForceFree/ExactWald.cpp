// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/ForceFree/ExactWald.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace ForceFree::Solutions {

ExactWald::ExactWald(const double magnetic_field_amplitude,
                     const Options::Context& context)
    : magnetic_field_amplitude_(magnetic_field_amplitude) {}

ExactWald::ExactWald(CkMigrateMessage* msg) : InitialData(msg) {}

std::unique_ptr<evolution::initial_data::InitialData> ExactWald::get_clone()
    const {
  return std::make_unique<ExactWald>(*this);
}

void ExactWald::pup(PUP::er& p) {
  InitialData::pup(p);
  p | background_spacetime_;
  p | magnetic_field_amplitude_;
}

PUP::able::PUP_ID ExactWald::my_PUP_ID = 0;

tuples::TaggedTuple<Tags::TildeE> ExactWald::variables(
    const tnsr::I<DataVector, 3>& coords, double /*t*/,
    tmpl::list<Tags::TildeE> /*meta*/) const {
  auto result = make_with_value<tnsr::I<DataVector, 3>>(coords, 0.0);

  DataVector r_squared = get(dot_product(coords, coords));
  const bool element_is_outside_horizon = min(r_squared) > 4.0;

  // matching radius
  const double r0 = 2.0;

  const auto& x = get<0>(coords);
  const auto& y = get<1>(coords);

  if (element_is_outside_horizon) {
    get<0>(result) = -y / r_squared;
    get<1>(result) = x / r_squared;
  } else {
    for (size_t i = 0; i < get_size(r_squared); ++i) {
      const auto& r_sq = get_element(r_squared, i);
      const auto& xi = get_element(x, i);
      const auto& yi = get_element(y, i);

      auto& TildeEx = get_element(get<0>(result), i);
      auto& TildeEy = get_element(get<1>(result), i);

      if (r_sq < square(r0)) {
        // inside horizon
        TildeEx = -yi / square(r0);
        TildeEy = xi / square(r0);
      } else {
        // outside horizon
        TildeEx = -yi / get_element(r_squared, i);
        TildeEy = xi / get_element(r_squared, i);
      }
    }
  }

  get<0>(result) *= 2.0 * magnetic_field_amplitude_;
  get<1>(result) *= 2.0 * magnetic_field_amplitude_;

  return result;
}

tuples::TaggedTuple<Tags::TildeB> ExactWald::variables(
    const tnsr::I<DataVector, 3>& cartesian_coords, double /*t*/,
    tmpl::list<Tags::TildeB> /*meta*/) const {
  auto tilde_b = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      cartesian_coords, 0.0);
  get<2>(tilde_b) = magnetic_field_amplitude_;

  return tilde_b;
}

tuples::TaggedTuple<Tags::TildePsi> ExactWald::variables(
    const tnsr::I<DataVector, 3>& x, double /*t*/,
    tmpl::list<Tags::TildePsi> /*meta*/) {
  return {make_with_value<Scalar<DataVector>>(x, 0.0)};
}

tuples::TaggedTuple<Tags::TildePhi> ExactWald::variables(
    const tnsr::I<DataVector, 3>& x, double /*t*/,
    tmpl::list<Tags::TildePhi> /*meta*/) {
  return {make_with_value<Scalar<DataVector>>(x, 0.0)};
}

tuples::TaggedTuple<Tags::TildeQ> ExactWald::variables(
    const tnsr::I<DataVector, 3>& x, double /*t*/,
    tmpl::list<Tags::TildeQ> /*meta*/) {
  return {make_with_value<Scalar<DataVector>>(x, 0.0)};
}

bool operator==(const ExactWald& lhs, const ExactWald& rhs) {
  return lhs.magnetic_field_amplitude_ == rhs.magnetic_field_amplitude_ and
         lhs.background_spacetime_ == rhs.background_spacetime_;
}

bool operator!=(const ExactWald& lhs, const ExactWald& rhs) {
  return not(lhs == rhs);
}

}  // namespace ForceFree::Solutions
