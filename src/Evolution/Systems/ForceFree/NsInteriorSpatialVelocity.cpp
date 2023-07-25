// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/NsInteriorSpatialVelocity.hpp"

#include <cstddef>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/AnalyticData/ForceFree/Factory.hpp"
#include "PointwiseFunctions/AnalyticSolutions/ForceFree/Factory.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"

namespace ForceFree::Tags {
namespace detail {

void ns_spatial_velocity_impl(
    const gsl::not_null<tnsr::I<DataVector, 3>*> ns_interior_spatial_velocity,
    const evolution::initial_data::InitialData& solution_or_data,
    const std::optional<Scalar<DataVector>>& neutron_star_interior_mask,
    [[maybe_unused]] const double time,
    const tnsr::I<DataVector, 3, Frame::Inertial>& inertial_coords) {
  const size_t num_grid_pts = get_size(get<0>(inertial_coords));
  set_number_of_grid_points(ns_interior_spatial_velocity, num_grid_pts);

  using all_solutions_and_data =
      tmpl::append<Solutions::all_solutions, AnalyticData::all_data>;

  call_with_dynamic_type<void, all_solutions_and_data>(
      &solution_or_data,
      [&ns_interior_spatial_velocity, &neutron_star_interior_mask,
       &inertial_coords, &num_grid_pts](const auto* initial_data_ptr) {
        using InitialData = std::decay_t<decltype(*initial_data_ptr)>;

        const auto set_everywhere_to_zero = [&ns_interior_spatial_velocity]() {
          get<0>(*ns_interior_spatial_velocity) = 0.0;
          get<1>(*ns_interior_spatial_velocity) = 0.0;
          get<2>(*ns_interior_spatial_velocity) = 0.0;
        };

        // Single star (pulsar) simulation
        if constexpr (std::is_same_v<InitialData,
                                     AnalyticData::RotatingDipole>) {
          if (neutron_star_interior_mask.has_value()) {
            const double omega = initial_data_ptr->angular_velocity();

            for (size_t m = 0; m < num_grid_pts; ++m) {
              if (get(neutron_star_interior_mask.value())[m] < 0.0) {
                // Interior
                // v^i = e^{ijk} Omega_j r_k
                get<0>(*ns_interior_spatial_velocity)[m] =
                    -omega * get<1>(inertial_coords)[m];
                get<1>(*ns_interior_spatial_velocity)[m] =
                    omega * get<0>(inertial_coords)[m];
                get<2>(*ns_interior_spatial_velocity)[m] = 0.0;
              } else {
                // Exterior
                get<0>(*ns_interior_spatial_velocity)[m] = 0.0;
                get<1>(*ns_interior_spatial_velocity)[m] = 0.0;
                get<2>(*ns_interior_spatial_velocity)[m] = 0.0;
              }
            }
          } else {
            set_everywhere_to_zero();
          }
        } else {
          set_everywhere_to_zero();
        }
      });
}
}  // namespace detail

}  // namespace ForceFree::Tags
