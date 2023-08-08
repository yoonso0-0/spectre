// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/Systems/ForceFree/System.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "PointwiseFunctions/AnalyticData/ForceFree/Factory.hpp"
#include "PointwiseFunctions/AnalyticData/ForceFree/RotatingDipole.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Utilities/TMPL.hpp"

namespace ForceFree {
/*!
 * \brief Impose the ideal MHD condition
 *
 */
struct ImposeMhdConditionInsideNs {
  using return_tags = tmpl::list<System::variables_tag>;

  using argument_tags =
      tmpl::list<evolution::initial_data::Tags::InitialData,
                 gr::Tags::SqrtDetSpatialMetric<DataVector>,
                 Tags::NsInteriorSpatialVelocity, Tags::NsInteriorMask>;

  static void apply(
      const gsl::not_null<System::variables_tag::type*> evolved_vars,
      const evolution::initial_data::InitialData& solution_or_data,
      const Scalar<DataVector>& sqrt_det_spatial_metric,
      const tnsr::I<DataVector, 3, Frame::Inertial>&
          ns_interior_spatial_velocity,
      const std::optional<Scalar<DataVector>>& ns_interior_mask);
};

}  // namespace ForceFree
