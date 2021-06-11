// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
#include "Utilities/Gsl.hpp"

namespace ScalarAdvection::Tags {
/*!
 * \brief Compute the advection velocity field of ScalarAdvection system
 */
template <size_t Dim>
struct VelocityFieldCompute {
  static void function(
      const gsl::not_null<tnsr::I<DataVector, Dim>*> velocity_field,
      const tnsr::I<DataVector, Dim, Frame::Inertial>&
          inertial_coords) noexcept;
};
}  // namespace ScalarAdvection::Tags
