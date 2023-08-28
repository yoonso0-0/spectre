// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DiscontinuousGalerkin/UsingSubcell.hpp"
#include "Evolution/Systems/ForceFree/System.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Utilities/Gsl.hpp"

namespace ForceFree {

template <typename Metavariables>
struct HeckInsideHorizon {
  using EvolvedVars = typename System::variables_tag::type;
  static constexpr size_t volume_dim = System::volume_dim;

  using return_tags = tmpl::list<System::variables_tag>;

  using argument_tags =
      tmpl::list<domain::Tags::Coordinates<3, Frame::Inertial>,
                 evolution::initial_data::Tags::InitialData>;

  static void apply(
      const gsl::not_null<EvolvedVars*> evolved_vars,
      const tnsr::I<DataVector, volume_dim, Frame::Inertial>& inertial_coords,
      const evolution::initial_data::InitialData& solution_or_data) {
    DataVector r_squared = get(dot_product(inertial_coords, inertial_coords));

    EvolvedVars initial_state_vars{};

    const size_t num_grid_pts = get_size(get<0>(inertial_coords));

    const bool element_is_outside_horizon = min(r_squared) > 4.0;

    if (element_is_outside_horizon) {
      (void)evolved_vars;  // avoid compiler warnings
      return;
    } else {
      initial_state_vars.initialize(num_grid_pts);

      using derived_classes =
          tmpl::at<typename Metavariables::factory_creation::factory_classes,
                   evolution::initial_data::InitialData>;

      call_with_dynamic_type<void, derived_classes>(
          &solution_or_data, [&initial_state_vars, &inertial_coords](
                                 const auto* const solution_or_data_ptr) {
            initial_state_vars.assign_subset(
                evolution::Initialization::initial_data(
                    *solution_or_data_ptr, inertial_coords, 0.0,
                    typename System::variables_tag::tags_list{}));
          });

      const auto& initial_tilde_e = get<Tags::TildeE>(initial_state_vars);
      const auto& initial_tilde_b = get<Tags::TildeB>(initial_state_vars);

      auto& tilde_e = get<Tags::TildeE>(*evolved_vars);
      auto& tilde_b = get<Tags::TildeB>(*evolved_vars);
      auto& tilde_phi = get<Tags::TildePhi>(*evolved_vars);
      auto& tilde_psi = get<Tags::TildePsi>(*evolved_vars);
      auto& tilde_q = get<Tags::TildeQ>(*evolved_vars);

      for (size_t m = 0; m < get_size(r_squared); ++m) {
        const double& r_sq = get_element(r_squared, m);

        if (r_sq < square(1.0)) {
          // inside horizon
          for (size_t d = 0; d < volume_dim; ++d) {
            get_element(tilde_e.get(d), m) =
                get_element(initial_tilde_e.get(d), m);
            get_element(tilde_b.get(d), m) =
                get_element(initial_tilde_b.get(d), m);
          }
          get_element(get(tilde_psi), m) = 0.0;
          get_element(get(tilde_phi), m) = 0.0;
          get_element(get(tilde_q), m) = 0.0;
        }
      }
    }
  }
};

namespace subcell {
template <typename Metavariables>
struct HeckInsideHorizon {
  using EvolvedVars = typename System::variables_tag::type;
  static constexpr size_t volume_dim = System::volume_dim;

  using return_tags = tmpl::list<System::variables_tag>;

  using argument_tags =
      tmpl::list<domain::Tags::Coordinates<3, Frame::Inertial>,
                 evolution::dg::subcell::Tags::Coordinates<3, Frame::Inertial>,
                 evolution::dg::subcell::Tags::ActiveGrid,
                 evolution::initial_data::Tags::InitialData>;

  static void apply(
      const gsl::not_null<EvolvedVars*> evolved_vars,
      const tnsr::I<DataVector, 3>& dg_inertial_coords,
      const tnsr::I<DataVector, 3>& subcell_inertial_coords,
      const evolution::dg::subcell::ActiveGrid& active_grid,
      const evolution::initial_data::InitialData& solution_or_data) {
    const auto& active_inertial_coords =
        [&dg_inertial_coords, &subcell_inertial_coords, &active_grid]() {
          if (active_grid == evolution::dg::subcell::ActiveGrid::Dg) {
            return dg_inertial_coords;
          } else {
            return subcell_inertial_coords;
          }
        }();

    DataVector r_squared =
        get(dot_product(active_inertial_coords, active_inertial_coords));

    EvolvedVars initial_state_vars{};

    const size_t num_grid_pts = get_size(get<0>(active_inertial_coords));

    const bool element_is_outside_horizon = min(r_squared) > 4.0;

    if (element_is_outside_horizon) {
      (void)evolved_vars;  // avoid compiler warnings
      return;
    } else {
      initial_state_vars.initialize(num_grid_pts);

      using derived_classes =
          tmpl::at<typename Metavariables::factory_creation::factory_classes,
                   evolution::initial_data::InitialData>;

      call_with_dynamic_type<void, derived_classes>(
          &solution_or_data, [&initial_state_vars, &active_inertial_coords](
                                 const auto* const solution_or_data_ptr) {
            initial_state_vars.assign_subset(
                evolution::Initialization::initial_data(
                    *solution_or_data_ptr, active_inertial_coords, 0.0,
                    typename System::variables_tag::tags_list{}));
          });

      const auto& initial_tilde_e = get<Tags::TildeE>(initial_state_vars);
      const auto& initial_tilde_b = get<Tags::TildeB>(initial_state_vars);
      auto& tilde_e = get<Tags::TildeE>(*evolved_vars);
      auto& tilde_b = get<Tags::TildeB>(*evolved_vars);

      for (size_t m = 0; m < get_size(r_squared); ++m) {
        const double& r_sq = get_element(r_squared, m);

        if (r_sq < 1.0) {  // inside horizon
          for (size_t d = 0; d < volume_dim; ++d) {
            get_element(tilde_e.get(d), m) =
                get_element(initial_tilde_e.get(d), m);
            get_element(tilde_b.get(d), m) =
                get_element(initial_tilde_b.get(d), m);
          }
        }
      }
    }
  }
};
}  // namespace subcell

}  // namespace ForceFree
