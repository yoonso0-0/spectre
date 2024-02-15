// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/Reconstruction.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/Imex/Tags/ImplicitHistory.hpp"
#include "Evolution/Imex/Tags/SolveFailures.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

namespace evolution::dg::subcell::detail {
/*!
 * \brief ee
 *
 *
 *
 */
struct ResizeImexTags {
  template <bool from_dg_to_fd, typename ImplicitSectors, typename DbTags,
            size_t Dim>
  static void apply(
      db::DataBox<DbTags>& box, const Mesh<Dim>& dg_mesh,
      const Mesh<Dim>& subcell_mesh,
      const evolution::dg::subcell::SubcellOptions& subcell_options) {
    tmpl::for_each<ImplicitSectors>([&](auto sector) {
      using implicit_sector = tmpl::type_from<std::decay_t<decltype(sector)>>;

      db::mutate<::imex::Tags::ImplicitHistory<implicit_sector>,
                 ::imex::Tags::SolveFailures<implicit_sector>>(
          [&](const auto implicit_history_ptr, const auto solve_failures_ptr) {
            if constexpr (from_dg_to_fd) {
              (void)subcell_options;  // avoid compiler warnings

              ASSERT(
                  implicit_history_ptr->size() > 0,
                  "We cannot have an empty history when unwinding, that's just "
                  "nutty. Did you call the action too early in the action "
                  "list?");

              implicit_history_ptr->undo_latest();
              implicit_history_ptr->map_entries([&dg_mesh, &subcell_mesh](
                                                    const auto entry) {
                *entry = fd::project(*entry, dg_mesh, subcell_mesh.extents());
              });

              set_number_of_grid_points(solve_failures_ptr,
                                        subcell_mesh.number_of_grid_points());
            } else {
              implicit_history_ptr->map_entries(
                  [&dg_mesh, &subcell_mesh,
                   &subcell_options](const auto entry) {
                    *entry = fd::reconstruct(
                        *entry, dg_mesh, subcell_mesh.extents(),
                        subcell_options.reconstruction_method());
                  });

              set_number_of_grid_points(solve_failures_ptr,
                                        dg_mesh.number_of_grid_points());
            }
          },
          make_not_null(&box));
    });
  }
};
}  // namespace evolution::dg::subcell::detail
