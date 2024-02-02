// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Protocols/Mutator.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/DidRollback.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace ForceFree::subcell {

struct ProjectElectricField : tt::ConformsTo<db::protocols::Mutator> {
  using return_tags = tmpl::list<ForceFree::Tags::TildeE>;

  using argument_tags =
      tmpl::list<ForceFree::Tags::TildeB,
                 gr::Tags::SpatialMetric<DataVector, 3>, domain::Tags::Mesh<3>,
                 evolution::dg::subcell::Tags::Mesh<3>,
                 evolution::dg::subcell::Tags::DidRollback,
                 evolution::dg::subcell::Tags::ActiveGrid>;

  static void apply(
      const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_e,
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
      const Mesh<3>& dg_mesh, const Mesh<3>& subcell_mesh,
      const bool did_rollback,
      const evolution::dg::subcell::ActiveGrid& active_grid);
};

}  // namespace ForceFree::subcell
