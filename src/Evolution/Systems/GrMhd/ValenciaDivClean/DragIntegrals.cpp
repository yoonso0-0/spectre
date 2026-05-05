// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/DragIntegrals.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Variables.hpp"
#include "PointwiseFunctions/GeneralRelativity/DerivativesOfSpacetimeMetric.hpp"
#include "PointwiseFunctions/Hydro/StressEnergy.hpp"

namespace grmhd::ValenciaDivClean {

void gravitational_drag_source_term(
    const gsl::not_null<tnsr::i<DataVector, 3>*> result,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& pressure,
    const Scalar<DataVector>& lorentz_factor, const Scalar<DataVector>& lapse,
    const Scalar<DataVector>& comoving_magnetic_field_magnitude,
    const tnsr::I<DataVector, 3>& spatial_velocity,
    const tnsr::I<DataVector, 3>& shift,
    const tnsr::I<DataVector, 3>& magnetic_field,
    const tnsr::ii<DataVector, 3>& spatial_metric,
    const tnsr::II<DataVector, 3>& inverse_spatial_metric,
    //
    const tnsr::i<DataVector, 3>& d_lapse,
    const tnsr::iJ<DataVector, 3>& d_shift,
    const tnsr::ijj<DataVector, 3>& d_spatial_metric) {
  TempBuffer<tmpl::list<::Tags::TempAA<0, 3>, ::Tags::Tempabb<0, 3>,
                        ::Tags::TempScalar<0>, ::Tags::TempI<0, 3>,
                        ::Tags::Tempii<0, 3>>>
      buffer{get(rest_mass_density).size()};

  auto& stress_energy_tensor = get<::Tags::TempAA<0, 3>>(buffer);
  auto& deriv_spacetime_metric = get<::Tags::Tempabb<0, 3>>(buffer);
  auto& dt_lapse = get<::Tags::TempScalar<0>>(buffer);
  auto& dt_shift = get<::Tags::TempI<0, 3>>(buffer);
  auto& dt_spatial_metric = get<::Tags::Tempii<0, 3>>(buffer);

  hydro::stress_energy_tensor(
      make_not_null(&stress_energy_tensor), rest_mass_density,
      specific_internal_energy, pressure, lorentz_factor, lapse,
      comoving_magnetic_field_magnitude, spatial_velocity, shift,
      magnetic_field, spatial_metric, inverse_spatial_metric);

  get(dt_lapse) = 0.;
  for (auto& component : dt_shift) {
    component = 0.;
  }
  for (auto& component : dt_spatial_metric) {
    component = 0.;
  }

  gr::derivatives_of_spacetime_metric(
      make_not_null(&deriv_spacetime_metric), lapse, dt_lapse, d_lapse, shift,
      dt_shift, d_shift, spatial_metric, dt_spatial_metric, d_spatial_metric);

  // compute T^{ab} d_i(g_ab) / 2
  for (size_t i = 0; i < 3; ++i) {
    result->get(i) = 0.;
    for (size_t j = 0; j < 4; ++j) {
      for (size_t k = 0; k < 4; ++k) {
        result->get(i) += stress_energy_tensor.get(j, k) *
                          deriv_spacetime_metric.get(i + 1, j, k);
      }
    }
    result->get(i) *= 0.5;
  }
}
}  // namespace grmhd::ValenciaDivClean
