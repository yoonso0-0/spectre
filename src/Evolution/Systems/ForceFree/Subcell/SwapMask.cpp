// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/Subcell/SwapMask.hpp"

#include <algorithm>
#include <cstddef>

#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"

#include <iostream>

namespace ForceFree::subcell {
void SwapMask::apply(const gsl::not_null<Scalar<DataVector>*> active_mask,
                     const gsl::not_null<Scalar<DataVector>*> inactive_mask,
                     const Mesh<3>& dg_mesh, const Mesh<3>& subcell_mesh,
                     evolution::dg::subcell::ActiveGrid active_grid) {
  const size_t active_mask_size = get(*active_mask).size();

  if (active_grid == evolution::dg::subcell::ActiveGrid::Dg) {
    // We might request a switch to the DG grid even if we are already on the DG
    // grid, and in this case we do nothing. This can occur when applying
    // SwapGrTags to a collection of elements that may have different TCI
    // results.
    if (active_mask_size != dg_mesh.number_of_grid_points()) {
      ASSERT(
          active_mask_size == subcell_mesh.number_of_grid_points(),
          "When swapping the GR variables from subcell to DG, the active "
          "GR variables should be holding the subcell variables and be of size "
              << subcell_mesh.number_of_grid_points()
              << " but they are of size " << active_mask->size());
      using std::swap;
      swap(*active_mask, *inactive_mask);
    }
  } else {
    if (active_mask_size != subcell_mesh.number_of_grid_points()) {
      ASSERT(active_mask_size == dg_mesh.number_of_grid_points(),
             "When swapping the GR variables from DG to subcell, the active "
             "GR variables should be holding the DG variables and be of size "
                 << dg_mesh.number_of_grid_points() << " but they are of size "
                 << active_mask->size());
      using std::swap;
      swap(*active_mask, *inactive_mask);
    }
  }
}
}  // namespace ForceFree::subcell
