// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/Subcell/SwapMask.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>

#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"

namespace ForceFree::subcell {

void SwapMask::apply(
    const gsl::not_null<std::optional<Scalar<DataVector>>*> active_mask,
    const gsl::not_null<std::optional<Scalar<DataVector>>*> inactive_mask,
    const Mesh<3>& dg_mesh, const Mesh<3>& subcell_mesh,
    evolution::dg::subcell::ActiveGrid active_grid) {
  ASSERT(not(active_mask->has_value() xor inactive_mask->has_value()),
         "Only one of active mask and inactive mask has value.");

  if (active_mask->has_value()) {
    const size_t active_mask_size = get(active_mask->value()).size();

    if (active_grid == evolution::dg::subcell::ActiveGrid::Dg) {
      // We might request a switch to the DG grid even if we are already on the
      // DG grid, and in this case we do nothing. This can occur when applying
      // SwapGrTags to a collection of elements that may have different TCI
      // results.
      if (active_mask_size != dg_mesh.number_of_grid_points()) {
        ASSERT(active_mask_size == subcell_mesh.number_of_grid_points(),
               "When swapping the GR variables from subcell to DG, the active "
               "GR variables should be holding the subcell variables and be of "
               "size "
                   << subcell_mesh.number_of_grid_points()
                   << " but they are of size " << active_mask_size);
        using std::swap;
        swap(active_mask->value(), inactive_mask->value());
      }
    } else if (active_grid == evolution::dg::subcell::ActiveGrid::Subcell) {
      if (active_mask_size != subcell_mesh.number_of_grid_points()) {
        ASSERT(active_mask_size == dg_mesh.number_of_grid_points(),
               "When swapping the GR variables from DG to subcell, the active "
               "GR variables should be holding the DG variables and be of size "
                   << dg_mesh.number_of_grid_points()
                   << " but they are of size " << active_mask);
        using std::swap;
        swap(active_mask->value(), inactive_mask->value());
      }
    }
  }
}

}  // namespace ForceFree::subcell
