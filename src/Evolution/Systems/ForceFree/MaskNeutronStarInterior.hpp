// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <type_traits>

#include "DataStructures/DataBox/Protocols/Mutator.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/DiscontinuousGalerkin/UsingSubcell.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/CreateIsCallable.hpp"

namespace ForceFree {
namespace detail {
CREATE_IS_CALLABLE(interior_mask)
CREATE_IS_CALLABLE_V(interior_mask)
CREATE_IS_CALLABLE_R_V(interior_mask)
}  // namespace detail

/*!
 * \brief Assign the masking scalar variable (see Tags::NsInteriorMask) at the
 * initialization phase in NS magnetosphere simulations.
 *
 * Run the `interior_mask()` member function of the initial data if it is
 * callable.
 */
template <typename Metavariables, bool UsingDgSubcell>
struct MaskNeutronStarInterior : tt::ConformsTo<db::protocols::Mutator> {
  using argument_tags = tmpl::flatten<tmpl::list<
      domain::Tags::Coordinates<3, Frame::Inertial>,
      tmpl::conditional_t<UsingDgSubcell,
                          tmpl::list<evolution::dg::subcell::Tags::Coordinates<
                                         3, Frame::Inertial>,
                                     evolution::dg::subcell::Tags::ActiveGrid>,
                          tmpl::list<>>,
      evolution::initial_data::Tags::InitialData>>;

  using return_tags = tmpl::flatten<
      tmpl::list<Tags::NsInteriorMask,
                 tmpl::conditional_t<UsingDgSubcell,
                                     evolution::dg::subcell::Tags::Inactive<
                                         Tags::NsInteriorMask>,
                                     tmpl::list<>>>>;

  using all_data_and_solutions =
      tmpl::at<typename Metavariables::factory_creation::factory_classes,
               evolution::initial_data::InitialData>;

  // The case `using_dg_subcell` == false
  static void apply(
      const gsl::not_null<std::optional<Scalar<DataVector>>*>
          dg_neutron_star_interior_mask,
      const tnsr::I<DataVector, 3, Frame::Inertial>& dg_inertial_coords,
      const evolution::initial_data::InitialData& solution_or_data) {
    call_with_dynamic_type<void, all_data_and_solutions>(
        &solution_or_data, [&dg_neutron_star_interior_mask,
                            &dg_inertial_coords](const auto* initial_data_ptr) {
          using InitialData = std::decay_t<decltype(*initial_data_ptr)>;

          if constexpr (detail::is_interior_mask_callable_r_v<
                            std::optional<Scalar<DataVector>>, InitialData,
                            tnsr::I<DataVector, 3, Frame::Inertial>>) {
            (*dg_neutron_star_interior_mask) =
                (*initial_data_ptr).interior_mask(dg_inertial_coords);
          }
        });
  }

  // The case `using_dg_subcell` == true
  //
  // Note the order of the returned arguments. Since executable is starting on
  // FD grid, mask on the DG grid should be matched with the Inactive<> tag.
  static void apply(
      const gsl::not_null<std::optional<Scalar<DataVector>>*> active_mask,
      const gsl::not_null<std::optional<Scalar<DataVector>>*> inactive_mask,
      const tnsr::I<DataVector, 3, Frame::Inertial>& dg_inertial_coords,
      const tnsr::I<DataVector, 3, Frame::Inertial>& fd_inertial_coords,
      const evolution::dg::subcell::ActiveGrid active_grid,
      const evolution::initial_data::InitialData& solution_or_data) {
    call_with_dynamic_type<void, all_data_and_solutions>(
        &solution_or_data,
        [&active_grid, &active_mask, &inactive_mask, &dg_inertial_coords,
         &fd_inertial_coords](const auto* initial_data_ptr) {
          using InitialData = std::decay_t<decltype(*initial_data_ptr)>;

          if constexpr (detail::is_interior_mask_callable_r_v<
                            std::optional<Scalar<DataVector>>, InitialData,
                            tnsr::I<DataVector, 3, Frame::Inertial>>) {
            if (active_grid == evolution::dg::subcell::ActiveGrid::Dg) {
              (*active_mask) =
                  (*initial_data_ptr).interior_mask(dg_inertial_coords);
              (*inactive_mask) =
                  (*initial_data_ptr).interior_mask(fd_inertial_coords);
            } else {
              (*inactive_mask) =
                  (*initial_data_ptr).interior_mask(dg_inertial_coords);
              (*active_mask) =
                  (*initial_data_ptr).interior_mask(fd_inertial_coords);
            }

            if (active_mask->has_value() xor inactive_mask->has_value()) {
              const size_t num_dg_pts = get<0>(dg_inertial_coords).size();
              const size_t num_fd_pts = get<0>(fd_inertial_coords).size();

              const size_t num_active_grid_pts =
                  active_grid == evolution::dg::subcell::ActiveGrid::Dg
                      ? num_dg_pts
                      : num_fd_pts;
              const size_t num_inactive_grid_pts =
                  active_grid == evolution::dg::subcell::ActiveGrid::Dg
                      ? num_fd_pts
                      : num_dg_pts;

              if (active_mask->has_value()) {
                (*inactive_mask) = Scalar<DataVector>{num_inactive_grid_pts};
                get((*inactive_mask).value()) = 1.0;
              } else {
                (*active_mask) = Scalar<DataVector>{num_active_grid_pts};
                get((*active_mask).value()) = 1.0;
              }
            }
          }
        });
  }
};

}  // namespace ForceFree
