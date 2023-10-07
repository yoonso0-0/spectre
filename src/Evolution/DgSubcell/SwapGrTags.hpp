// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>

#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg::subcell {
/*!
 * \brief Swaps the inactive and active background GR variables.
 *
 * The values on the subcells are at the cell-centers.
 *
 * It should be possible to reduce memory usage by deallocating the GR variables
 * on the DG grid when switching to subcell. However, the opposite case is not
 * true since the GR variables are needed on the subcells if a neighbor is using
 * subcell in order to compute the neighbor's fluxes.
 *
 * \note The `active_grid` is the grid we are swapping to, which may be the same
 * as the current grid. On output the `active_gr_vars` will match the grid that
 * `active_grid` is. This mutator is a no-op if they matched on input.
 */
template <typename System>
struct SwapGrTags {
  using return_tags = tmpl::list<typename System::spacetime_variables_tag,
                                 evolution::dg::subcell::Tags::Inactive<
                                     typename System::spacetime_variables_tag>>;
  using argument_tags =
      tmpl::list<::domain::Tags::Mesh<3>, evolution::dg::subcell::Tags::Mesh<3>,
                 evolution::dg::subcell::Tags::ActiveGrid>;

  static void apply(
      gsl::not_null<
          Variables<typename System::spacetime_variables_tag::tags_list>*>
          active_gr_vars,
      gsl::not_null<typename evolution::dg::subcell::Tags::Inactive<
          typename System::spacetime_variables_tag>::type*>
          inactive_gr_vars,
      const Mesh<3>& dg_mesh, const Mesh<3>& subcell_mesh,
      evolution::dg::subcell::ActiveGrid active_grid) {
    if (active_grid == evolution::dg::subcell::ActiveGrid::Dg) {
      // We might request a switch to the DG grid even if we are already on the
      // DG grid, and in this case we do nothing. This can occur when applying
      // SwapGrTags to a collection of elements that may have different TCI
      // results.
      if (active_gr_vars->number_of_grid_points() !=
          dg_mesh.number_of_grid_points()) {
        ASSERT(active_gr_vars->number_of_grid_points() ==
                   subcell_mesh.number_of_grid_points(),
               "When swapping the GR variables from subcell to DG, the active "
               "GR variables should be holding the subcell variables and be of "
               "size "
                   << subcell_mesh.number_of_grid_points()
                   << " but they are of size "
                   << active_gr_vars->number_of_grid_points());
        using std::swap;
        swap(*active_gr_vars, *inactive_gr_vars);
      }
    } else {
      if (active_gr_vars->number_of_grid_points() !=
          subcell_mesh.number_of_grid_points()) {
        ASSERT(active_gr_vars->number_of_grid_points() ==
                   dg_mesh.number_of_grid_points(),
               "When swapping the GR variables from DG to subcell, the active "
               "GR variables should be holding the DG variables and be of size "
                   << dg_mesh.number_of_grid_points()
                   << " but they are of size "
                   << active_gr_vars->number_of_grid_points());
        using std::swap;
        swap(*active_gr_vars, *inactive_gr_vars);
      }
    }
  }
};
}  // namespace evolution::dg::subcell
