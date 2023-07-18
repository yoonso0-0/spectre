// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/ForceFree/MagnetosphericWald.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "Options/Options.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/TaggedTuple.hpp"

#include <iostream>

namespace ForceFree::AnalyticData {

MagnetosphericWald::MagnetosphericWald(const double spin,
                                       const Options::Context& context)
    : spin_(spin),
      background_spacetime_{1.0, {{0.0, 0.0, spin_}}, {{0.0, 0.0, 0.0}}},
      kerr_schild_coords_{1.0, spin_} {
  if (abs(spin_) >= 1.0) {
    PARSE_ERROR(context, "The dimensionless spin ("
                             << spin_ << ") must be smaller than 1.0");
  }
}

MagnetosphericWald::MagnetosphericWald(CkMigrateMessage* msg)
    : InitialData(msg) {}

std::unique_ptr<evolution::initial_data::InitialData>
MagnetosphericWald::get_clone() const {
  return std::make_unique<MagnetosphericWald>(*this);
}

void MagnetosphericWald::pup(PUP::er& p) {
  InitialData::pup(p);
  p | spin_;
  p | background_spacetime_;
  p | kerr_schild_coords_;
}

PUP::able::PUP_ID MagnetosphericWald::my_PUP_ID = 0;

tnsr::I<DataVector, 3, Frame::Inertial> MagnetosphericWald::regularize_coords(
    const tnsr::I<DataVector, 3>& inertial_coords) const {
  auto regularized_coords =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(inertial_coords,
                                                               0.0);
  for (size_t d = 0; d < 3; ++d) {
    regularized_coords.get(d) = inertial_coords.get(d);
  }

  DataVector r_squared =
      get(kerr_schild_coords_.r_coord_squared(inertial_coords));
  DataVector spherical_radius_squared =
      get(dot_product(inertial_coords, inertial_coords));
  const bool element_is_outside_heck_region =
      min(spherical_radius_squared) > 1.0;

  if (element_is_outside_heck_region) {
    return inertial_coords;
  } else {
    const size_t num_grid_pts = get_size(get<0>(inertial_coords));

    const double sign_x = std::copysign(
        1.0, get_element(get<0>(inertial_coords), 0) +
                 get_element(get<0>(inertial_coords), num_grid_pts - 1));
    const double sign_y = std::copysign(
        1.0, get_element(get<1>(inertial_coords), 0) +
                 get_element(get<1>(inertial_coords), num_grid_pts - 1));
    const double sign_z = std::copysign(
        1.0, get_element(get<2>(inertial_coords), 0) +
                 get_element(get<2>(inertial_coords), num_grid_pts - 1));

    DataVector z_squared = square(get<2>(inertial_coords));
    DataVector varphi_squared = spherical_radius_squared - z_squared;

    for (size_t m = 0; m < num_grid_pts; ++m) {
      const double& spherical_r_sq = get_element(spherical_radius_squared, m);
      const double& r_sq = get_element(r_squared, m);

      // Heck the coordinate & physical singularity on the disk x^2 + y^2 = a^2

      const double delta = 1e-2;

      const bool inside_singularity =
          get_element(varphi_squared, m) <= square(spin_);
      const bool close_to_xy_plane = get_element(z_squared, m) < square(delta);

      if (inside_singularity and close_to_xy_plane) {
        get_element(get<2>(regularized_coords), m) = sign_z * delta;
      }
    }
  }

  return regularized_coords;
}

tuples::TaggedTuple<Tags::TildeE> MagnetosphericWald::variables(
    const tnsr::I<DataVector, 3>& x, tmpl::list<Tags::TildeE> /*meta*/) {
  return {make_with_value<tnsr::I<DataVector, 3>>(x, 0.0)};
}

tuples::TaggedTuple<Tags::TildeB> MagnetosphericWald::variables(
    const tnsr::I<DataVector, 3>& cartesian_coords,
    tmpl::list<Tags::TildeB> /*meta*/) const {
  auto tilde_b = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      cartesian_coords, 0.0);

  // FIXME : make this as member variables? option?
  const double mass = 1.0;
  const double B0 = 1.0;
  const double a_squared = square(spin_);

  const auto& x = get<0>(cartesian_coords);
  const auto& y = get<1>(cartesian_coords);
  const auto& z = get<2>(cartesian_coords);

  // FIXME : combine this into a single allocation
  DataVector r_squared =
      get(kerr_schild_coords_.r_coord_squared(cartesian_coords));
  DataVector r_coords = sqrt(r_squared);
  DataVector z_squared = square(z);
  DataVector prefactor_1 = (r_squared - a_squared) / (r_squared + a_squared);
  DataVector temp = square(r_squared) + square(spin_) * z_squared;

  for (size_t m = 0; m < get_size(r_squared); ++m) {
    if (UNLIKELY(r_squared[m] < 1.0e-15)) {
      get<0>(tilde_b)[m] = 0.0;
      get<1>(tilde_b)[m] = 0.0;
      get<2>(tilde_b)[m] = 1.0;
    } else {
      const double temp2 = get_element(r_squared, m) *
                           get_element(r_coords, m) * get_element(z, m);

      get_element(get<0>(tilde_b), m) =
          2.0 * spin_ * mass * B0 * get_element(prefactor_1, m) * temp2 *
          (spin_ * x[m] - r_coords[m] * y[m]) / square(get_element(temp, m));

      get_element(get<1>(tilde_b), m) =
          2.0 * spin_ * mass * B0 * get_element(prefactor_1, m) * temp2 *
          (r_coords[m] * x[m] + spin_ * y[m]) / square(get_element(temp, m));

      get_element(get<2>(tilde_b), m) =
          1.0 - 2 * a_squared * mass * get_element(r_coords, m) *
                    (get_element(r_squared, m) + get_element(z_squared, m)) /
                    ((get_element(r_squared, m) + a_squared) *
                     (square(r_squared)[m] + a_squared * z_squared[m]));
    }
  }

  get<2>(tilde_b) *= B0;

  return tilde_b;
}

tuples::TaggedTuple<Tags::TildePsi> MagnetosphericWald::variables(
    const tnsr::I<DataVector, 3>& x, tmpl::list<Tags::TildePsi> /*meta*/) {
  return {make_with_value<Scalar<DataVector>>(x, 0.0)};
}

tuples::TaggedTuple<Tags::TildePhi> MagnetosphericWald::variables(
    const tnsr::I<DataVector, 3>& x, tmpl::list<Tags::TildePhi> /*meta*/) {
  return {make_with_value<Scalar<DataVector>>(x, 0.0)};
}

tuples::TaggedTuple<Tags::TildeQ> MagnetosphericWald::variables(
    const tnsr::I<DataVector, 3>& x, tmpl::list<Tags::TildeQ> /*meta*/) {
  return {make_with_value<Scalar<DataVector>>(x, 0.0)};
}

}  // namespace ForceFree::AnalyticData
