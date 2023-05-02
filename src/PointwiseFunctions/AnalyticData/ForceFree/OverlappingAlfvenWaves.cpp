// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/ForceFree/OverlappingAlfvenWaves.hpp"

#include <cstddef>
#include <memory>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Options/ParseError.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace ForceFree::AnalyticData {

OverlappingAlfvenWaves::OverlappingAlfvenWaves(
    const double normalized_wave_amplitude, const Options::Context& context)
    : normalized_wave_amplitude_(normalized_wave_amplitude) {
  if (normalized_wave_amplitude_ <= 0.0) {
    PARSE_ERROR(context, "The normalized wave amplitude dB / B ("
                             << normalized_wave_amplitude_
                             << ") must be positive.");
  }
}

OverlappingAlfvenWaves::OverlappingAlfvenWaves(CkMigrateMessage* msg)
    : InitialData(msg) {}

std::unique_ptr<evolution::initial_data::InitialData>
OverlappingAlfvenWaves::get_clone() const {
  return std::make_unique<OverlappingAlfvenWaves>(*this);
}

void OverlappingAlfvenWaves::pup(PUP::er& p) {
  InitialData::pup(p);
  p | normalized_wave_amplitude_;
}

PUP::able::PUP_ID OverlappingAlfvenWaves::my_PUP_ID = 0;

tuples::TaggedTuple<Tags::TildeE> OverlappingAlfvenWaves::variables(
    const tnsr::I<DataVector, 3>& coords,
    tmpl::list<Tags::TildeE> /*meta*/) const {
  auto result = make_with_value<tnsr::I<DataVector, 3>>(coords, 0.0);

  const auto& x = get<0>(coords);
  const auto& y = get<1>(coords);
  const auto& z = get<2>(coords);

  get<0>(result) =
      -normalized_wave_amplitude_ * cos(k_perp_ * x + k_parallel_ * z);
  get<1>(result) =
      normalized_wave_amplitude_ * cos(k_perp_ * y - k_parallel_ * z);

  return result;
}

tuples::TaggedTuple<Tags::TildeB> OverlappingAlfvenWaves::variables(
    const tnsr::I<DataVector, 3>& coords,
    tmpl::list<Tags::TildeB> /*meta*/) const {
  auto result = make_with_value<tnsr::I<DataVector, 3>>(coords, 0.0);

  const auto& x = get<0>(coords);
  const auto& y = get<1>(coords);
  const auto& z = get<2>(coords);

  get<0>(result) =
      normalized_wave_amplitude_ * cos(k_perp_ * y - k_parallel_ * z);
  get<1>(result) =
      -normalized_wave_amplitude_ * cos(k_perp_ * x + k_parallel_ * z);
  get<2>(result) = 1.0;

  return result;
}

tuples::TaggedTuple<Tags::TildePsi> OverlappingAlfvenWaves::variables(
    const tnsr::I<DataVector, 3>& coords, tmpl::list<Tags::TildePsi> /*meta*/) {
  return {make_with_value<Scalar<DataVector>>(coords, 0.0)};
}

tuples::TaggedTuple<Tags::TildePhi> OverlappingAlfvenWaves::variables(
    const tnsr::I<DataVector, 3>& coords, tmpl::list<Tags::TildePhi> /*meta*/) {
  return {make_with_value<Scalar<DataVector>>(coords, 0.0)};
}

tuples::TaggedTuple<Tags::TildeQ> OverlappingAlfvenWaves::variables(
    const tnsr::I<DataVector, 3>& coords,
    tmpl::list<Tags::TildeQ> /*meta*/) const {
  auto result = make_with_value<Scalar<DataVector>>(coords, 0.0);

  const auto& x = get<0>(coords);
  const auto& y = get<1>(coords);
  const auto& z = get<2>(coords);

  get(result) =
      normalized_wave_amplitude_ * k_perp_ *
      (sin(k_perp_ * x + k_parallel_ * z) - sin(k_perp_ * y - k_parallel_ * z));

  return result;
}

bool operator==(const OverlappingAlfvenWaves& lhs,
                const OverlappingAlfvenWaves& rhs) {
  return lhs.normalized_wave_amplitude_ == rhs.normalized_wave_amplitude_ and
         lhs.background_spacetime_ == rhs.background_spacetime_;
}

bool operator!=(const OverlappingAlfvenWaves& lhs,
                const OverlappingAlfvenWaves& rhs) {
  return not(lhs == rhs);
}

}  // namespace ForceFree::AnalyticData
