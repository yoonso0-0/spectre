// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/GrMhd/BhlAccretion.hpp"

#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace grmhd::Solutions {

BhlAccretion::BhlAccretion(const double velocity, const double density,
                           const double pressure, const double adiabatic_index,
                           const double spin, const Options::Context& context)
    : velocity_(velocity),
      density_(density),
      pressure_(pressure),
      spin_(spin),
      adiabatic_index_(adiabatic_index),
      equation_of_state_(adiabatic_index),
      background_spacetime_{1.0, {{0.0, 0.0, spin_}}, {{0.0, 0.0, 0.0}}} {}

std::unique_ptr<evolution::initial_data::InitialData> BhlAccretion::get_clone()
    const {
  return std::make_unique<BhlAccretion>(*this);
}

BhlAccretion::BhlAccretion(CkMigrateMessage* msg) : InitialData(msg) {}

void BhlAccretion::pup(PUP::er& p) {
  InitialData::pup(p);
  p | velocity_;
  p | density_;
  p | pressure_;
  p | adiabatic_index_;
  p | spin_;
  p | equation_of_state_;
  p | background_spacetime_;
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>>
BhlAccretion::variables(
    const tnsr::I<DataType, 3>& x, double /*t*/,
    tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/) const {
  return {make_with_value<Scalar<DataType>>(x, density_)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::ElectronFraction<DataType>>
BhlAccretion::variables(
    const tnsr::I<DataType, 3>& x, double /*t*/,
    tmpl::list<hydro::Tags::ElectronFraction<DataType>> /*meta*/) const {
  return {make_with_value<Scalar<DataType>>(x, 0.5)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>>
BhlAccretion::variables(
    const tnsr::I<DataType, 3>& x, double /*t*/,
    tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/) const {
  return {make_with_value<Scalar<DataType>>(
      x, pressure_ / ((adiabatic_index_ - 1.0) * density_))};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::Pressure<DataType>> BhlAccretion::variables(
    const tnsr::I<DataType, 3>& x, double /*t*/,
    tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/) const {
  return {make_with_value<Scalar<DataType>>(x, pressure_)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DataType, 3>>
BhlAccretion::variables(
    const tnsr::I<DataType, 3>& x, double /*t*/,
    tmpl::list<hydro::Tags::SpatialVelocity<DataType, 3>> /*meta*/) const {
  auto result = make_with_value<tnsr::I<DataType, 3>>(x, 0.0);
  result.get(0) = velocity_;
  return {std::move(result)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3>>
BhlAccretion::variables(
    const tnsr::I<DataType, 3>& x, double /*t*/,
    tmpl::list<hydro::Tags::MagneticField<DataType, 3>> /*meta*/) const {
  return {make_with_value<tnsr::I<DataType, 3>>(x, 0.0)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>
BhlAccretion::variables(
    const tnsr::I<DataType, 3>& x, double /*t*/,
    tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/) const {
  return {make_with_value<Scalar<DataType>>(x, 0.0)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>>
BhlAccretion::variables(
    const tnsr::I<DataType, 3>& x, double /*t*/,
    tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/) const {
  return {make_with_value<Scalar<DataType>>(
      x, 1.0 / sqrt(1.0 - square(velocity_)))};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DataType>>
BhlAccretion::variables(
    const tnsr::I<DataType, 3>& x, double t,
    tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>> /*meta*/) const {
  Scalar<DataType> specific_internal_energy = std::move(
      get<hydro::Tags::SpecificInternalEnergy<DataType>>(variables<DataType>(
          x, t, tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>>{})));
  get(specific_internal_energy) *= adiabatic_index_;
  get(specific_internal_energy) += 1.0;
  return {std::move(specific_internal_energy)};
}

PUP::able::PUP_ID BhlAccretion::my_PUP_ID = 0;

bool operator==(const BhlAccretion& lhs, const BhlAccretion& rhs) {
  // there is no comparison operator for the EoS, but should be okay as
  // the adiabatic_indexs are compared
  return lhs.velocity_ == rhs.velocity_ and lhs.density_ == rhs.density_ and
         lhs.adiabatic_index_ == rhs.adiabatic_index_ and
         lhs.spin_ == rhs.spin_ and
         lhs.background_spacetime_ == rhs.background_spacetime_;
}

bool operator!=(const BhlAccretion& lhs, const BhlAccretion& rhs) {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define TAG(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE_SCALARS(_, data)                                      \
  template tuples::TaggedTuple < TAG(data) < DTYPE(data) >>               \
      BhlAccretion::variables(const tnsr::I<DTYPE(data), 3>& x, double t, \
                              tmpl::list < TAG(data) < DTYPE(data) >>     \
                              /*meta*/) const;

GENERATE_INSTANTIATIONS(
    INSTANTIATE_SCALARS, (double, DataVector),
    (hydro::Tags::RestMassDensity, hydro::Tags::ElectronFraction,
     hydro::Tags::SpecificInternalEnergy, hydro::Tags::Pressure,
     hydro::Tags::DivergenceCleaningField, hydro::Tags::LorentzFactor,
     hydro::Tags::SpecificEnthalpy))

#define INSTANTIATE_VECTORS(_, data)                                           \
  template tuples::TaggedTuple < TAG(data) < DTYPE(data),                      \
      3 >> BhlAccretion::variables(const tnsr::I<DTYPE(data), 3>& x, double t, \
                                   tmpl::list < TAG(data) < DTYPE(data), 3 >>  \
                                   /*meta*/) const;

GENERATE_INSTANTIATIONS(INSTANTIATE_VECTORS, (double, DataVector),
                        (hydro::Tags::SpatialVelocity,
                         hydro::Tags::MagneticField))

#undef DTYPE
#undef TAG
#undef INSTANTIATE_SCALARS
#undef INSTANTIATE_VECTORS
}  // namespace grmhd::Solutions
