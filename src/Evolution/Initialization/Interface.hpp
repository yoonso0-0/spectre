// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Initialization/MergeIntoDataBox.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
/// \endcond

namespace Initialization {
namespace Actions {
/// \brief Initialize items related to the interfaces between Elements and on
/// external boundaries
///
/// DataBox changes:
/// - Adds:
///   * `face_tags<Tags::InternalDirections<Dim>>`
///   * `face_tags<Tags::BoundaryDirectionsInterior<Dim>>`
///   * `face_tags<Tags::BoundaryDirectionsExterior<Dim>>`
///
/// - For face_tags:
///   * `Tags::InterfaceComputeItem<Directions, Tags::Direction<Dim>>`
///   * `Tags::InterfaceComputeItem<Directions, Tags::InterfaceMesh<Dim>>`
///   * `Tags::Slice<Directions, typename System::variables_tag>`
///   * `Tags::InterfaceComputeItem<Directions,
///                                Tags::UnnormalizedFaceNormal<Dim>>`
///   * Tags::InterfaceComputeItem<Directions,
///                                typename System::template magnitude_tag<
///                                    Tags::UnnormalizedFaceNormal<Dim>>>
///   * `Tags::InterfaceComputeItem<
///         Directions, Tags::Normalized<Tags::UnnormalizedFaceNormal<Dim>>>`
/// - Removes: nothing
/// - Modifies: nothing
template <typename System>
struct Interface {
  static constexpr size_t dim = System::volume_dim;
  using simple_tags = db::AddSimpleTags<::Tags::Interface<
      ::Tags::BoundaryDirectionsExterior<dim>, typename System::variables_tag>>;

  template <typename Directions>
  using face_tags = tmpl::list<
      Directions,
      ::Tags::InterfaceComputeItem<Directions, ::Tags::Direction<dim>>,
      ::Tags::InterfaceComputeItem<Directions, ::Tags::InterfaceMesh<dim>>,
      ::Tags::Slice<Directions, typename System::variables_tag>,
      ::Tags::Slice<Directions, typename System::spacetime_variables_tag>,
      ::Tags::Slice<Directions, typename System::primitive_variables_tag>,
      ::Tags::InterfaceComputeItem<Directions,
                                   ::Tags::UnnormalizedFaceNormal<dim>>,
      ::Tags::InterfaceComputeItem<Directions,
                                   typename System::template magnitude_tag<
                                       ::Tags::UnnormalizedFaceNormal<dim>>>,
      ::Tags::InterfaceComputeItem<
          Directions,
          ::Tags::NormalizedCompute<::Tags::UnnormalizedFaceNormal<dim>>>>;

  using ext_tags = tmpl::list<
      ::Tags::BoundaryDirectionsExterior<dim>,
      ::Tags::InterfaceComputeItem<::Tags::BoundaryDirectionsExterior<dim>,
                                   ::Tags::Direction<dim>>,
      ::Tags::InterfaceComputeItem<::Tags::BoundaryDirectionsExterior<dim>,
                                   ::Tags::InterfaceMesh<dim>>,
      ::Tags::Slice<::Tags::BoundaryDirectionsExterior<dim>,
                    typename System::spacetime_variables_tag>,
      ::Tags::Slice<::Tags::BoundaryDirectionsExterior<dim>,
                    typename System::primitive_variables_tag>,
      ::Tags::InterfaceComputeItem<::Tags::BoundaryDirectionsExterior<dim>,
                                   ::Tags::BoundaryCoordinates<dim>>,
      ::Tags::InterfaceComputeItem<::Tags::BoundaryDirectionsExterior<dim>,
                                   ::Tags::UnnormalizedFaceNormal<dim>>,
      ::Tags::InterfaceComputeItem<::Tags::BoundaryDirectionsExterior<dim>,
                                   typename System::template magnitude_tag<
                                       ::Tags::UnnormalizedFaceNormal<dim>>>,
      ::Tags::InterfaceComputeItem<
          ::Tags::BoundaryDirectionsExterior<dim>,
          ::Tags::NormalizedCompute<::Tags::UnnormalizedFaceNormal<dim>>>>;

  using compute_tags =
      tmpl::append<face_tags<::Tags::InternalDirections<dim>>,
                   face_tags<::Tags::BoundaryDirectionsInterior<dim>>,
                   ext_tags>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const auto& mesh = db::get<::Tags::Mesh<dim>>(box);
    std::unordered_map<Direction<dim>,
                       db::item_type<typename System::variables_tag>>
        external_boundary_vars{};

    for (const auto& direction :
         db::get<::Tags::Element<dim>>(box).external_boundaries()) {
      external_boundary_vars[direction] =
          db::item_type<typename System::variables_tag>{
              mesh.slice_away(direction.dimension()).number_of_grid_points()};
    }

    return std::make_tuple(
        merge_into_databox<Interface, simple_tags, compute_tags>(
            std::move(box), std::move(external_boundary_vars)));
  }
};
}  // namespace Actions
}  // namespace Initialization
