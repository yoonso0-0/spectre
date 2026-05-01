// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/GrMhd/BondiHoyleLyttletonAccretion.hpp"

#include <array>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Options/Context.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace grmhd::AnalyticData {

BondiHoyleLyttletonAccretion::BondiHoyleLyttletonAccretion(
    const double mass, const std::array<double, 3>& spin,
    const std::array<double, 3>& center, const double adiabatic_index,
    const double sound_speed, const double wind_velocity,
    const Options::Context& /*context*/)
    : adiabatic_index_(adiabatic_index),
      sound_speed_(sound_speed),
      wind_velocity_(wind_velocity),
      equation_of_state_{adiabatic_index_},
      background_spacetime_{mass, spin, center} {
  pressure_ =
      square(sound_speed_) * (adiabatic_index_ - 1.0) /
      (adiabatic_index_ * (adiabatic_index_ - 1.0 - square(sound_speed_)));
}

std::unique_ptr<evolution::initial_data::InitialData>
BondiHoyleLyttletonAccretion::get_clone() const {
  return std::make_unique<BondiHoyleLyttletonAccretion>(*this);
}

BondiHoyleLyttletonAccretion::BondiHoyleLyttletonAccretion(
    CkMigrateMessage* msg)
    : InitialData(msg) {}

void BondiHoyleLyttletonAccretion::pup(PUP::er& p) {
  InitialData::pup(p);
  p | adiabatic_index_;
  p | sound_speed_;
  p | wind_velocity_;
  p | pressure_;
  p | equation_of_state_;
  p | background_spacetime_;
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>>
BondiHoyleLyttletonAccretion::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/) const {
  return {make_with_value<Scalar<DataType>>(x, 1.0)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::ElectronFraction<DataType>>
BondiHoyleLyttletonAccretion::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<hydro::Tags::ElectronFraction<DataType>> /*meta*/) const {
  return {make_with_value<Scalar<DataType>>(x, 0.1)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>>
BondiHoyleLyttletonAccretion::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/) const {
  return equation_of_state_.specific_internal_energy_from_density_and_pressure(
      get<hydro::Tags::RestMassDensity<DataType>>(
          variables(x, tmpl::list<hydro::Tags::RestMassDensity<DataType>>{})),
      get<hydro::Tags::Pressure<DataType>>(
          variables(x, tmpl::list<hydro::Tags::Pressure<DataType>>{})));
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::Pressure<DataType>>
BondiHoyleLyttletonAccretion::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/) const {
  return {make_with_value<Scalar<DataType>>(x, pressure_)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DataType, 3>>
BondiHoyleLyttletonAccretion::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<hydro::Tags::SpatialVelocity<DataType, 3>> /*meta*/) const {
  auto result = make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.0);

  // FIXME
  // Use a more detailed initial condition for spatial velocity. Also check the
  // convention
  //
  get<0>(result) = -wind_velocity_;
  return {std::move(result)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3>>
BondiHoyleLyttletonAccretion::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<hydro::Tags::MagneticField<DataType, 3>> /*meta*/) const {
  return {make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.0)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>
BondiHoyleLyttletonAccretion::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/) const {
  return {make_with_value<Scalar<DataType>>(x, 0.0)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>>
BondiHoyleLyttletonAccretion::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/) const {
  // FIXME
  // We might want to compute Lorentz factor consistently, using spacetime
  // metric.
  //
  return {make_with_value<Scalar<DataType>>(
      x, 1.0 / sqrt(1.0 - square(wind_velocity_)))};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DataType>>
BondiHoyleLyttletonAccretion::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>> /*meta*/) const {
  return hydro::relativistic_specific_enthalpy(
      get<hydro::Tags::RestMassDensity<DataType>>(
          variables(x, tmpl::list<hydro::Tags::RestMassDensity<DataType>>{})),
      get<hydro::Tags::SpecificInternalEnergy<DataType>>(variables(
          x, tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>>{})),
      get<hydro::Tags::Pressure<DataType>>(
          variables(x, tmpl::list<hydro::Tags::Pressure<DataType>>{})));
}

PUP::able::PUP_ID BondiHoyleLyttletonAccretion::my_PUP_ID = 0;

bool operator==(const BondiHoyleLyttletonAccretion& lhs,
                const BondiHoyleLyttletonAccretion& rhs) {
  return lhs.background_spacetime_ == rhs.background_spacetime_ and
         lhs.adiabatic_index_ == rhs.adiabatic_index_ and
         lhs.sound_speed_ == rhs.sound_speed_ and
         lhs.wind_velocity_ == rhs.wind_velocity_;
}

bool operator!=(const BondiHoyleLyttletonAccretion& lhs,
                const BondiHoyleLyttletonAccretion& rhs) {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define TAG(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE_SCALARS(_, data)                         \
  template tuples::TaggedTuple < TAG(data) < DTYPE(data) >>  \
      BondiHoyleLyttletonAccretion::variables(               \
          const tnsr::I<DTYPE(data), 3, Frame::Inertial>& x, \
          tmpl::list < TAG(data) < DTYPE(data) >>            \
          /*meta*/) const;

GENERATE_INSTANTIATIONS(
    INSTANTIATE_SCALARS, (double, DataVector),
    (hydro::Tags::RestMassDensity, hydro::Tags::ElectronFraction,
     hydro::Tags::SpecificInternalEnergy, hydro::Tags::Pressure,
     hydro::Tags::DivergenceCleaningField, hydro::Tags::LorentzFactor,
     hydro::Tags::SpecificEnthalpy))

#define INSTANTIATE_VECTORS(_, data)                             \
  template tuples::TaggedTuple < TAG(data) < DTYPE(data), 3,     \
      Frame::Inertial >>                                         \
          BondiHoyleLyttletonAccretion::variables(               \
              const tnsr::I<DTYPE(data), 3, Frame::Inertial>& x, \
              tmpl::list < TAG(data) < DTYPE(data), 3 >>         \
              /*meta*/) const;

GENERATE_INSTANTIATIONS(INSTANTIATE_VECTORS, (double, DataVector),
                        (hydro::Tags::SpatialVelocity,
                         hydro::Tags::MagneticField))

#undef DTYPE
#undef TAG
#undef INSTANTIATE_SCALARS
#undef INSTANTIATE_VECTORS

}  // namespace grmhd::AnalyticData
