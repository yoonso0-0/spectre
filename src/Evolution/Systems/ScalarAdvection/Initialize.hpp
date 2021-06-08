// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Initialization/DiscontinuousGalerkin.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Systems/ScalarAdvection/System.hpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
#include "Evolution/Systems/ScalarAdvection/VelocityField.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"

namespace ScalarAdvection {
namespace Actions {
/*!
 * \ingroup InitializationGroup
 * \brief Initialize items related to the ScalarAdvection system
 *
 * Add advection velocity field to the evolution databox
 *
 * DataBox changes:
 *  - Adds:
 *    `ScalarAdvection::Tags::VelocityField`
 *  - Removes: nothing
 *  - Modifies: nothing
 *
 */

template <size_t Dim>
struct InitializeVelocityField {
  using simple_tags = tmpl::list<ScalarAdvection::Tags::VelocityField<Dim>>;

  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    // retrieve mesh and coordinates
    const auto& mesh = db::get<domain::Tags::Mesh<Dim>>(box);
    const auto& inertial_coords =
        get<domain::Tags::Coordinates<Dim, Frame::Inertial>>(box);

    // initialize velocity field
    tnsr::I<DataVector, Dim, Frame::Inertial> velocity_field{
        mesh.number_of_grid_points(), 0.};
    ScalarAdvection::Tags::VelocityFieldCompute<Dim>::function(
        make_not_null(&velocity_field), inertial_coords);

    Initialization::mutate_assign<simple_tags>(make_not_null(&box),
                                               std::move(velocity_field));
    return std::make_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace ScalarAdvection
