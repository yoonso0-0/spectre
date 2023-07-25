// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace ForceFree {
namespace Tags {
namespace detail {
void ns_spatial_velocity_impl(
    const gsl::not_null<tnsr::I<DataVector, 3>*> ns_interior_spatial_velocity,
    const evolution::initial_data::InitialData& solution_or_data,
    const std::optional<Scalar<DataVector>>& neutron_star_interior_mask,
    [[maybe_unused]] const double time,
    const tnsr::I<DataVector, 3, Frame::Inertial>& inertial_coords);
}

/*!
 * \brief efefef
 *
 * \note unlike other tensors, we just resize this tag every time.. since it's
 * not very expensive to recompute?
 */
template <bool UsingDgSubcell>
struct NsInteriorSpatialVelocityCompute : NsInteriorSpatialVelocity,
                                          db::ComputeTag {
  using argument_tags = tmpl::append<
      tmpl::list<evolution::initial_data::Tags::InitialData, ::Tags::Time,
                 NsInteriorMask, domain::Tags::Coordinates<3, Frame::Inertial>>,
      tmpl::conditional_t<UsingDgSubcell,
                          tmpl::list<evolution::dg::subcell::Tags::Coordinates<
                                         3, Frame::Inertial>,
                                     evolution::dg::subcell::Tags::ActiveGrid>,
                          tmpl::list<>>>;

  using return_type = NsInteriorSpatialVelocity::type;
  using base = NsInteriorSpatialVelocity;

  // DG-only
  static void function(
      gsl::not_null<tnsr::I<DataVector, 3>*> ns_interior_spatial_velocity,
      const evolution::initial_data::InitialData& solution_or_data,
      const double time,
      const std::optional<Scalar<DataVector>>& neutron_star_interior_mask,
      const tnsr::I<DataVector, 3>& dg_inertial_coords) {
    detail::ns_spatial_velocity_impl(
        ns_interior_spatial_velocity, solution_or_data,
        neutron_star_interior_mask, time, dg_inertial_coords);
  }

  // DG-FD
  static void function(
      gsl::not_null<tnsr::I<DataVector, 3>*> ns_interior_spatial_velocity,
      const evolution::initial_data::InitialData& solution_or_data,
      const double time,
      const std::optional<Scalar<DataVector>>& neutron_star_interior_mask,
      const tnsr::I<DataVector, 3>& dg_inertial_coords,
      const tnsr::I<DataVector, 3>& subcell_inertial_coords,
      const evolution::dg::subcell::ActiveGrid& active_grid) {
    const auto call_impl =
        [&solution_or_data, &ns_interior_spatial_velocity, &time,
         &neutron_star_interior_mask](const auto active_inertial_coords) {
          detail::ns_spatial_velocity_impl(
              ns_interior_spatial_velocity, solution_or_data,
              neutron_star_interior_mask, time, active_inertial_coords);
        };

    if (active_grid == evolution::dg::subcell::ActiveGrid::Dg) {
      call_impl(dg_inertial_coords);
    } else {
      call_impl(subcell_inertial_coords);
    }
  }
};

}  // namespace Tags
}  // namespace ForceFree
