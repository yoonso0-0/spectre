// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/ForceFree/OrbitingBinary.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TaggedTuple.hpp"

#include <iostream>

namespace ForceFree::AnalyticData {

OrbitingBinary::OrbitingBinary(const double angular_velocity_one,
                               const double angular_velocity_two,
                               const double orbital_radius,
                               const Options::Context& context)
    : angular_velocity_one_(angular_velocity_one),
      angular_velocity_two_(angular_velocity_two),
      orbital_radius_(orbital_radius) {
  if (angular_velocity_one >= 1.0) {
    PARSE_ERROR(context, "The rotation angular velocity one ("
                             << angular_velocity_one_
                             << ") must be between 0.0 and 1.0");
  }
  if (angular_velocity_two >= 1.0) {
    PARSE_ERROR(context, "The rotation angular velocity two ("
                             << angular_velocity_two_
                             << ") must be between 0.0 and 1.0");
  }
  if (orbital_radius <= 1.0) {
    PARSE_ERROR(context, "The orbital radius (" << orbital_radius_
                                                << ") must be larger than 1.0");
  }
}

OrbitingBinary::OrbitingBinary(CkMigrateMessage* msg) : InitialData(msg) {}

std::unique_ptr<evolution::initial_data::InitialData>
OrbitingBinary::get_clone() const {
  return std::make_unique<OrbitingBinary>(*this);
}

void OrbitingBinary::pup(PUP::er& p) {
  InitialData::pup(p);
  p | angular_velocity_one_;
  p | angular_velocity_two_;
  p | orbital_radius_;
  p | background_spacetime_;
}

PUP::able::PUP_ID OrbitingBinary::my_PUP_ID = 0;

tuples::TaggedTuple<Tags::TildeE> OrbitingBinary::variables(
    const tnsr::I<DataVector, 3>& coords, tmpl::list<Tags::TildeE> /*meta*/) {
  return {make_with_value<tnsr::I<DataVector, 3>>(coords, 0.0)};
}

tuples::TaggedTuple<Tags::TildeB> OrbitingBinary::variables(
    const tnsr::I<DataVector, 3>& coords,
    tmpl::list<Tags::TildeB> /*meta*/) const {
  auto result = make_with_value<tnsr::I<DataVector, 3>>(coords, 0.0);

  const double delta_ = 0.4;
  const double tilt_angle_ = 0.0;

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
  get<0>(result) =
      cos_alpha * get<0>(tilde_b_prime) + sin_alpha * get<2>(tilde_b_prime);
  get<1>(result) = get<1>(tilde_b_prime);
  get<2>(result) =
      -sin_alpha * get<0>(tilde_b_prime) + cos_alpha * get<2>(tilde_b_prime);

  return result;
}

tuples::TaggedTuple<Tags::TildePsi> OrbitingBinary::variables(
    const tnsr::I<DataVector, 3>& coords, tmpl::list<Tags::TildePsi> /*meta*/) {
  return {make_with_value<Scalar<DataVector>>(coords, 0.0)};
}

tuples::TaggedTuple<Tags::TildePhi> OrbitingBinary::variables(
    const tnsr::I<DataVector, 3>& coords, tmpl::list<Tags::TildePhi> /*meta*/) {
  return {make_with_value<Scalar<DataVector>>(coords, 0.0)};
}

tuples::TaggedTuple<Tags::TildeQ> OrbitingBinary::variables(
    const tnsr::I<DataVector, 3>& coords, tmpl::list<Tags::TildeQ> /*meta*/) {
  return {make_with_value<Scalar<DataVector>>(coords, 0.0)};
}

std::optional<Scalar<DataVector>> compute_mask(
    const tnsr::I<DataVector, 3>& coords, const double ns_radius_squared,
    const double center_location_x) {
  std::optional<Scalar<DataVector>> mask{};

  auto displacement = make_with_value<tnsr::I<DataVector, 3>>(coords, 0.0);
  get<0>(displacement) = get<0>(coords) + center_location_x;
  get<1>(displacement) = get<1>(coords);
  get<2>(displacement) = get<2>(coords);
  const DataVector d_squared = get(dot_product(displacement, displacement));

  const size_t num_grid_points = get<0>(coords).size();

  if (min(d_squared) < ns_radius_squared) {
    mask = Scalar<DataVector>{num_grid_points};  // Allocate the mask vector
    for (size_t i = 0; i < num_grid_points; ++i) {
      if (d_squared[i] < ns_radius_squared) {
        get(mask.value())[i] = -1.0;
      } else {
        get(mask.value())[i] = +1.0;
      }
    }
  }
  return mask;
}

std::optional<Scalar<DataVector>> OrbitingBinary::interior_mask_one(
    const tnsr::I<DataVector, 3, Frame::Inertial>& x) const {
  const double ns_radius_squared = 1.0;
  return compute_mask(x, ns_radius_squared, -orbital_radius_);
}
std::optional<Scalar<DataVector>> OrbitingBinary::interior_mask_two(
    const tnsr::I<DataVector, 3, Frame::Inertial>& x) const {
  const double ns_radius_squared = 1.0;
  return compute_mask(x, ns_radius_squared, orbital_radius_);
}

std::optional<Scalar<DataVector>> OrbitingBinary::interior_mask(
    const tnsr::I<DataVector, 3>& x) const {
  std::optional<Scalar<DataVector>> result{};

  const double ns_radius_squared = 1.0;
  const auto mask_one = compute_mask(x, ns_radius_squared, -orbital_radius_);
  const auto mask_two = compute_mask(x, ns_radius_squared, orbital_radius_);

  if (mask_one.has_value() or mask_two.has_value()) {
    result = Scalar<DataVector>{get<0>(x).size()};

    // std::cout << "Mask 1 : " << mask_one << std::endl;
    // std::cout << "Mask 2 : " << mask_two << std::endl;

    if (mask_one.has_value() and mask_two.has_value()) {
      get(result.value()) = max(get(mask_one.value()), get(mask_two.value()));
    } else if ((not mask_one.has_value()) and mask_two.has_value()) {
      result.value() = mask_two.value();
    } else if (mask_one.has_value() and (not mask_two.has_value())) {
      result.value() = mask_one.value();
    }
  }

  return result;
}

bool operator==(const OrbitingBinary& lhs, const OrbitingBinary& rhs) {
  return lhs.angular_velocity_one_ == rhs.angular_velocity_one_ and
         lhs.angular_velocity_two_ == rhs.angular_velocity_two_ and
         lhs.orbital_radius_ == rhs.orbital_radius_ and
         lhs.background_spacetime_ == rhs.background_spacetime_;
}

bool operator!=(const OrbitingBinary& lhs, const OrbitingBinary& rhs) {
  return not(lhs == rhs);
}

}  // namespace ForceFree::AnalyticData
