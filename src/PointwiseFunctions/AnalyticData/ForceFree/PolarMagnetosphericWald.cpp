// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/ForceFree/PolarMagnetosphericWald.hpp"

#include <pup.h>
#include <utility>

namespace ForceFree::AnalyticData {

PolarMagnetosphericWald::PolarMagnetosphericWald(
    grmhd::AnalyticData::SphericalTorus torus_map)
    : torus_map_(std::move(torus_map)) {}

std::unique_ptr<evolution::initial_data::InitialData>
PolarMagnetosphericWald::get_clone() const {
  return std::make_unique<PolarMagnetosphericWald>(*this);
}

PolarMagnetosphericWald::PolarMagnetosphericWald(CkMigrateMessage* msg)
    : magnetospheric_wald_(msg) {}

void PolarMagnetosphericWald::pup(PUP::er& p) {
  p | torus_map_;
}

PUP::able::PUP_ID PolarMagnetosphericWald::my_PUP_ID = 0;  // NOLINT

bool operator==(const PolarMagnetosphericWald& lhs,
                const PolarMagnetosphericWald& rhs) {
  return lhs.magnetospheric_wald_ == rhs.magnetospheric_wald_ and
         lhs.torus_map_ == rhs.torus_map_;
}

bool operator!=(const PolarMagnetosphericWald& lhs,
                const PolarMagnetosphericWald& rhs) {
  return not(lhs == rhs);
}
}  // namespace ForceFree::AnalyticData
