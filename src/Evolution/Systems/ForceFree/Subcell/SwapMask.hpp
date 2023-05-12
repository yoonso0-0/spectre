// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>

#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace ForceFree::subcell {

/*!
 * \brief Swap (why)
 *
 */
struct SwapMask {
  using return_tags =
      tmpl::list<Tags::NsInteriorMask,
                 evolution::dg::subcell::Tags::Inactive<Tags::NsInteriorMask>>;
  using argument_tags =
      tmpl::list<::domain::Tags::Mesh<3>, evolution::dg::subcell::Tags::Mesh<3>,
                 evolution::dg::subcell::Tags::ActiveGrid>;

  static void apply(
      gsl::not_null<std::optional<Scalar<DataVector>>*> active_mask,
      gsl::not_null<std::optional<Scalar<DataVector>>*> inactive_mask,
      const Mesh<3>& dg_mesh, const Mesh<3>& subcell_mesh,
      evolution::dg::subcell::ActiveGrid active_grid);
};

}  // namespace ForceFree::subcell
