// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/ForceFree/RotatingDipoleBoundary.hpp"

#include <cstddef>
#include <memory>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace ForceFree::Solutions {

RotatingDipoleBoundary::RotatingDipoleBoundary(
    const double vector_potential_amplitude, const double varpi0,
    const double delta, const double angular_velocity, const double tilt_angle,
    const Options::Context& context)
    : vector_potential_amplitude_(vector_potential_amplitude),
      varpi0_(varpi0),
      delta_(delta),
      angular_velocity_(angular_velocity),
      tilt_angle_(tilt_angle) {
  if (varpi0 < 0.0) {
    PARSE_ERROR(context, "The length constant varpi0 ("
                             << varpi0_ << ") cannot be negative");
  }
  if (delta < 0.0) {
    PARSE_ERROR(context,
                "The small number delta (" << delta_ << ") cannot be negative");
  }
  if (abs(angular_velocity) >= 1.0) {
    PARSE_ERROR(context, "The rotation angular velocity ("
                             << angular_velocity_
                             << ") must be between -1.0 and 1.0");
  }
  if ((tilt_angle < 0.0) or (tilt_angle > M_PI)) {
    PARSE_ERROR(context, "The rotator tilt angle ("
                             << tilt_angle_ << ") must be between 0 and Pi");
  }
}

RotatingDipoleBoundary::RotatingDipoleBoundary(CkMigrateMessage* msg)
    : InitialData(msg) {}

std::unique_ptr<evolution::initial_data::InitialData>
RotatingDipoleBoundary::get_clone() const {
  return std::make_unique<RotatingDipoleBoundary>(*this);
}

void RotatingDipoleBoundary::pup(PUP::er& p) {
  InitialData::pup(p);
  p | vector_potential_amplitude_;
  p | varpi0_;
  p | delta_;
  p | angular_velocity_;
  p | tilt_angle_;
  p | background_spacetime_;
}

PUP::able::PUP_ID RotatingDipoleBoundary::my_PUP_ID = 0;

tuples::TaggedTuple<Tags::TildeE> RotatingDipoleBoundary::variables(
    const tnsr::I<DataVector, 3>& coords, double t,
    tmpl::list<Tags::TildeE> /*meta*/) const {
  auto result = make_with_value<tnsr::I<DataVector, 3>>(coords, 0.0);

  auto velocity = make_with_value<tnsr::I<DataVector, 3>>(coords, 0.0);
  get<0>(velocity) = -angular_velocity_ * get<1>(coords);
  get<1>(velocity) = angular_velocity_ * get<0>(coords);

  const auto tilde_b =
      get<Tags::TildeB>(variables(coords, t, tmpl::list<Tags::TildeB>{}));

  const auto& bx = get<0>(tilde_b);
  const auto& by = get<1>(tilde_b);
  const auto& bz = get<2>(tilde_b);
  const auto& vx = get<0>(velocity);
  const auto& vy = get<1>(velocity);
  const auto& vz = get<2>(velocity);

  get<0>(result) = (by * vz - bz * vy) / 1.0;
  get<1>(result) = (bz * vx - bx * vz) / 1.0;
  get<2>(result) = (bx * vy - by * vx) / 1.0;

  return result;
}

tuples::TaggedTuple<Tags::TildeB> RotatingDipoleBoundary::variables(
    const tnsr::I<DataVector, 3>& coords, double t,
    tmpl::list<Tags::TildeB> /*meta*/) const {
  auto initial_b_field = make_with_value<tnsr::I<DataVector, 3>>(coords, 0.0);
  auto result = make_with_value<tnsr::I<DataVector, 3>>(coords, 0.0);

  const double sin_alpha = sin(tilt_angle_);
  const double cos_alpha = cos(tilt_angle_);

  // Coordinates and magnetic fields in the tilted axis
  const auto& x = get<0>(coords);
  const auto& y = get<1>(coords);
  const auto& z = get<2>(coords);
  const DataVector x_prime = cos_alpha * x - sin_alpha * z;
  const DataVector z_prime = sin_alpha * x + cos_alpha * z;

  auto tilde_b_prime = make_with_value<tnsr::I<DataVector, 3>>(coords, 0.0);

  // Regularized dipole field
  const DataVector r_squared = get(dot_product(coords, coords));
  const DataVector one_over_radius_factor =
      1.0 / pow<5>(sqrt(r_squared + square(delta_)));
  get<0>(tilde_b_prime) = 3.0 * x_prime * z_prime * one_over_radius_factor;
  get<1>(tilde_b_prime) = 3.0 * y * z_prime * one_over_radius_factor;
  get<2>(tilde_b_prime) =
      (3.0 * square(z_prime) - r_squared + 2.0 * square(delta_)) *
      one_over_radius_factor;

  // Rotation
  get<0>(initial_b_field) =
      cos_alpha * get<0>(tilde_b_prime) + sin_alpha * get<2>(tilde_b_prime);
  get<1>(initial_b_field) = get<1>(tilde_b_prime);
  get<2>(initial_b_field) =
      -sin_alpha * get<0>(tilde_b_prime) + cos_alpha * get<2>(tilde_b_prime);

  // Time-dependence
  const double phi_angle = angular_velocity_ * t;
  const double sin_phi = sin(phi_angle);
  const double cos_phi = cos(phi_angle);

  get<0>(result) =
      get<0>(initial_b_field) * cos_phi - get<1>(initial_b_field) * sin_phi;
  get<1>(result) =
      get<0>(initial_b_field) * sin_phi + get<1>(initial_b_field) * cos_phi;
  get<2>(result) = get<2>(initial_b_field);

  return result;
}

tuples::TaggedTuple<Tags::TildePsi> RotatingDipoleBoundary::variables(
    const tnsr::I<DataVector, 3>& coords, double /*t*/,
    tmpl::list<Tags::TildePsi> /*meta*/) {
  return {make_with_value<Scalar<DataVector>>(coords, 0.0)};
}

tuples::TaggedTuple<Tags::TildePhi> RotatingDipoleBoundary::variables(
    const tnsr::I<DataVector, 3>& coords, double /*t*/,
    tmpl::list<Tags::TildePhi> /*meta*/) {
  return {make_with_value<Scalar<DataVector>>(coords, 0.0)};
}

tuples::TaggedTuple<Tags::TildeQ> RotatingDipoleBoundary::variables(
    const tnsr::I<DataVector, 3>& coords, double /*t*/,
    tmpl::list<Tags::TildeQ> /*meta*/) {
  return {make_with_value<Scalar<DataVector>>(coords, 0.0)};
}

bool operator==(const RotatingDipoleBoundary& lhs,
                const RotatingDipoleBoundary& rhs) {
  return lhs.vector_potential_amplitude_ == rhs.vector_potential_amplitude_ and
         lhs.varpi0_ == rhs.varpi0_ and lhs.delta_ == rhs.delta_ and
         lhs.angular_velocity_ == rhs.angular_velocity_ and
         lhs.tilt_angle_ == rhs.tilt_angle_ and
         lhs.background_spacetime_ == rhs.background_spacetime_;
}

bool operator!=(const RotatingDipoleBoundary& lhs,
                const RotatingDipoleBoundary& rhs) {
  return not(lhs == rhs);
}

}  // namespace ForceFree::Solutions
