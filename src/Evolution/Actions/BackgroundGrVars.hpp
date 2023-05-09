// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Block.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/BlockId.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Initialization/InitialData.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Time/Tags.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace evolution::Actions {

/*!
 * \brief Allocate or assign background general relativity quantities needed
 * for evolution systems that run on a fixed spacetime e.g. ValenciaDivclean,
 * ForceFree.
 *
 * If a template parameter `used_in_evolution_phase` == true, this action
 * mutates (updates) the value of background metric variables in box.
 *
 * Uses:
 *  - DataBox:
 *    * `evolution::initial_data::Tags::InitialData` or
 *       `::Tags::AnalyticSolutionOrData`
 *    * `domain::Tags::Coordinates<Dim, Frame::Inertial>`
 *    * `domain::Tags::Element<Dim>`
 *    * `domain::Tags::Domain<Dim>`
 *    * `::Tags::Time`
 *
 * DataBox changes:
 *  - Adds: nothing
 *  - Removes : nothing
 *  - Modifies :
 *    * system::spacetime_variables_tag
 *    * system::flux_spacetime_variables_tag
 *    * system::inverse_spatial_metric_tag
 *
 * If a template parameter `used_in_evolution_phase` == false, this action can
 * be used for initialization. Allocate and assign the values of background
 * metric variables into DataBox.
 *
 * Uses:
 *  - DataBox:
 *    * `evolution::initial_data::Tags::InitialData` or
 *       `::Tags::AnalyticSolutionOrData`
 *    * `domain::Tags::Coordinates<Dim, Frame::Inertial>`
 *    * `domain::Tags::Mesh<Dim>`
 *    * `::Tags::Time`
 *
 * DataBox changes:
 *  - Adds:
 *    * system::spacetime_variables_tag
 *    * system::flux_spacetime_variables_tag
 *    * system::inverse_spatial_metric_tag
 *  - Removes : nothing
 *  - Modifies : nothing
 *
 */
template <typename System, bool used_in_evolution_phase>
struct BackgroundGrVars {
  using simple_tags_from_options = tmpl::list<::Tags::Time>;

  // Collect all the GR quantities used in the templated evolution system
  using gr_variables_tag = ::Tags::Variables<tmpl::remove_duplicates<
      tmpl::append<typename System::spacetime_variables_tag::tags_list,
                   typename System::flux_spacetime_variables_tag::tags_list,
                   tmpl::list<typename System::inverse_spatial_metric_tag>>>>;

  using simple_tags = tmpl::list<gr_variables_tag>;
  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    // Note : once every evolution system is refactored to use the runtime
    // initial data, we can just use
    // `evolution::initial_data::Tags::Initialdata` to retrieve the analytic
    // solution or data. Then we may be able to convert this action into a
    // mutator or an argument of Initialization::Actions::AddSimpleTags<>.
    //
    if constexpr (db::tag_is_retrievable_v<
                      evolution::initial_data::Tags::InitialData,
                      db::DataBox<DbTagsList>>) {
      using derived_classes =
          tmpl::at<typename Metavariables::factory_creation::factory_classes,
                   evolution::initial_data::InitialData>;
      call_with_dynamic_type<void, derived_classes>(
          &db::get<evolution::initial_data::Tags::InitialData>(box),
          [&box](const auto* const data_or_solution) {
            impl(make_not_null(&box), *data_or_solution);
          });
    } else if constexpr (db::tag_is_retrievable_v<
                             ::Tags::AnalyticSolutionOrData,
                             db::DataBox<DbTagsList>>) {
      impl(make_not_null(&box), db::get<::Tags::AnalyticSolutionOrData>(box));
    } else {
      ERROR(
          "Either ::Tags::AnalyticSolutionOrData or "
          "evolution::initial_data::Tags::InitialData must be in the "
          "DataBox.");
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }

 private:
  template <typename DbTagsList, typename T>
  static void impl(const gsl::not_null<db::DataBox<DbTagsList>*> box,
                   const T& solution_or_data) {
    static constexpr size_t Dim = System::volume_dim;

    using GrVars = typename gr_variables_tag::type;

    if constexpr (used_in_evolution_phase) {
      // Check if the mesh is actually moving i.e. block coordinate map is
      // time-dependent. If not, we can skip the evaluation of GR variables
      // since they may stay with their values assigned at the initialization
      // phase.
      const auto& element_id = get<domain::Tags::Element<Dim>>(*box).id();
      const size_t block_id = element_id.block_id();
      const Block<Dim>& block =
          get<domain::Tags::Domain<Dim>>(*box).blocks()[block_id];

      if (block.is_time_dependent()) {
        const double time = db::get<::Tags::Time>(*box);
        const auto& inertial_coords =
            db::get<::domain::Tags::Coordinates<Dim, Frame::Inertial>>(*box);

        db::mutate<gr_variables_tag>(
            box,
            [&time, &inertial_coords, &solution_or_data](
                const gsl::not_null<typename gr_variables_tag::type*> gr_vars) {
              gr_vars->assign_subset(evolution::Initialization::initial_data(
                  solution_or_data, inertial_coords, time,
                  typename GrVars::tags_list{}));
            });
      }
    } else {
      // We need an allocation of GR variables only in the initialization phase.
      const double initial_time = db::get<::Tags::Time>(*box);
      const auto& inertial_coords =
          db::get<::domain::Tags::Coordinates<Dim, Frame::Inertial>>(*box);
      const size_t num_grid_points =
          db::get<::domain::Tags::Mesh<Dim>>(*box).number_of_grid_points();
      GrVars gr_vars{num_grid_points};
      gr_vars.assign_subset(evolution::Initialization::initial_data(
          solution_or_data, inertial_coords, initial_time,
          typename GrVars::tags_list{}));

      ::Initialization::mutate_assign<simple_tags>(box, std::move(gr_vars));
    }
  }
};

}  // namespace evolution::Actions
