// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/ImposeMhdConditionInsideNs.hpp"

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

namespace ForceFree {

void ImposeMhdConditionInsideNs::apply(
    const gsl::not_null<System::variables_tag::type*> evolved_vars,
    const evolution::initial_data::InitialData& solution_or_data,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::I<DataVector, 3, Frame::Inertial>& ns_interior_spatial_velocity,
    const std::optional<Scalar<DataVector>>& ns_interior_mask) {
  using all_solutions_and_data =
      tmpl::append<Solutions::all_solutions, AnalyticData::all_data>;

  using neutron_star_id_list = tmpl::list<AnalyticData::RotatingDipole>;

  call_with_dynamic_type<void, all_solutions_and_data>(
      &solution_or_data, [&](const auto* initial_data_ptr) {
        using InitialData = std::decay_t<decltype(*initial_data_ptr)>;

        if constexpr (tmpl::list_contains_v<neutron_star_id_list,
                                            InitialData>) {
          const size_t num_grid_pts = get_size(get(sqrt_det_spatial_metric));

          ASSERT(num_grid_pts == (*evolved_vars).number_of_grid_points(),
                 " Number of grid points ("
                     << num_grid_pts
                     << ") does not match the size of evolved variables ("
                     << (*evolved_vars).number_of_grid_points() << ") ");
          ASSERT(
              num_grid_pts == get<0>(ns_interior_spatial_velocity).size(),
              " Number of grid points ("
                  << num_grid_pts
                  << ") does not match the size of NS interior velocity field ("
                  << get(ns_interior_mask.value()).size() << ") ");

          if (not ns_interior_mask.has_value()) {
            (void)evolved_vars;  // avoid compiler warnings
          } else {
            ASSERT(num_grid_pts == get(ns_interior_mask.value()).size(),
                   " Number of grid points ("
                       << num_grid_pts
                       << ") does not match the size of NS interior mask ("
                       << get(ns_interior_mask.value()).size() << ") ");

            auto& tilde_e = get<Tags::TildeE>(*evolved_vars);
            auto& tilde_b = get<Tags::TildeB>(*evolved_vars);
            auto& tilde_psi = get<Tags::TildePsi>(*evolved_vars);
            auto& tilde_q = get<Tags::TildeQ>(*evolved_vars);

            for (size_t i = 0; i < num_grid_pts; ++i) {
              if (get(ns_interior_mask.value())[i] < 0.0) {
                // interior of the NS -> overwrite with MHD condition

                // FIXME : to be more correct, we need to use one-forms of v^i
                // and B^i fields..

                const auto& bx = get<0>(tilde_b)[i];
                const auto& by = get<1>(tilde_b)[i];
                const auto& bz = get<2>(tilde_b)[i];

                const auto& vx = get<0>(ns_interior_spatial_velocity)[i];
                const auto& vy = get<1>(ns_interior_spatial_velocity)[i];
                const auto& vz = get<2>(ns_interior_spatial_velocity)[i];

                const auto& sqrt_gamma = get(sqrt_det_spatial_metric)[i];

                get<0>(tilde_e)[i] = (by * vz - bz * vy) / sqrt_gamma;
                get<1>(tilde_e)[i] = (bz * vx - bx * vz) / sqrt_gamma;
                get<2>(tilde_e)[i] = (bx * vy - by * vx) / sqrt_gamma;

                get(tilde_psi)[i] = 0.0;
                get(tilde_q)[i] = 0.0;
              }
            }
          }
        }
      });
}
}  // namespace ForceFree
