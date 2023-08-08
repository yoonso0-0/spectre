// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/ElectromagneticVariables.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/LeviCivitaIterator.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

namespace ForceFree {

void em_field_from_evolved_fields(
    const gsl::not_null<tnsr::I<DataVector, 3>*> vector,
    const tnsr::I<DataVector, 3>& densitized_vector,
    const Scalar<DataVector>& sqrt_det_spatial_metric) {
  get<0>(*vector) = get<0>(densitized_vector) / get(sqrt_det_spatial_metric);
  get<1>(*vector) = get<1>(densitized_vector) / get(sqrt_det_spatial_metric);
  get<2>(*vector) = get<2>(densitized_vector) / get(sqrt_det_spatial_metric);
}

void charge_density_from_tilde_q(
    const gsl::not_null<Scalar<DataVector>*> charge_density,
    const Scalar<DataVector>& tilde_q,
    const Scalar<DataVector>& sqrt_det_spatial_metric) {
  get(*charge_density) = get(tilde_q) / get(sqrt_det_spatial_metric);
}

void electric_current_density_from_tilde_j(
    const gsl::not_null<tnsr::I<DataVector, 3>*> electric_current_density,
    const tnsr::I<DataVector, 3>& tilde_j,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const Scalar<DataVector>& lapse) {
  em_field_from_evolved_fields(electric_current_density, tilde_j,
                               sqrt_det_spatial_metric);
  get<0>(*electric_current_density) =
      get<0>(*electric_current_density) / get(lapse);
  get<1>(*electric_current_density) =
      get<1>(*electric_current_density) / get(lapse);
  get<2>(*electric_current_density) =
      get<2>(*electric_current_density) / get(lapse);
}

void joule_heating(const gsl::not_null<Scalar<DataVector>*> joule_heating,
                   const tnsr::I<DataVector, 3>& tilde_e,
                   const tnsr::I<DataVector, 3>& tilde_j,
                   const Scalar<DataVector>& lapse,
                   const Scalar<DataVector>& sqrt_det_spatial_metric) {
  dot_product(joule_heating, tilde_e, tilde_j);
  get(*joule_heating) /= get(lapse) * square(get(sqrt_det_spatial_metric));
}

void electromagnetic_energy_density(
    const gsl::not_null<Scalar<DataVector>*> electromagnetic_energy_density,
    const tnsr::I<DataVector, 3>& tilde_e,
    const tnsr::I<DataVector, 3>& tilde_b,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric) {
  Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>>> buffer{
      get<0>(tilde_e).size()};
  auto& tilde_e_squared = get<::Tags::TempScalar<0>>(buffer);
  auto& tilde_b_squared = get<::Tags::TempScalar<1>>(buffer);

  dot_product(make_not_null(&tilde_e_squared), tilde_e, tilde_e,
              spatial_metric);
  dot_product(make_not_null(&tilde_b_squared), tilde_b, tilde_b,
              spatial_metric);

  get(*electromagnetic_energy_density) =
      0.5 * (get(tilde_e_squared) + get(tilde_b_squared)) /
      square(get(sqrt_det_spatial_metric));
}

void poynting_covector(
    const gsl::not_null<tnsr::i<DataVector, 3>*> poynting_covector,
    const tnsr::I<DataVector, 3>& tilde_e,
    const tnsr::I<DataVector, 3>& tilde_b,
    const Scalar<DataVector>& sqrt_det_spatial_metric) {
  const size_t num_grid_pts = get(sqrt_det_spatial_metric).size();

  set_number_of_grid_points(poynting_covector, num_grid_pts);
  get<0>(*poynting_covector) = 0.0;
  get<1>(*poynting_covector) = 0.0;
  get<2>(*poynting_covector) = 0.0;

  for (LeviCivitaIterator<3> it; it; ++it) {
    const auto& i = it[0];
    const auto& j = it[1];
    const auto& k = it[2];
    (*poynting_covector).get(i) += it.sign() * tilde_e.get(j) * tilde_b.get(k) /
                                   get(sqrt_det_spatial_metric);
  }
}

void poynting_flux(const gsl::not_null<Scalar<DataVector>*> poynting_flux,
                   const tnsr::i<DataVector, 3>& poynting_covector,
                   const tnsr::I<DataVector, 3>& normal_vector) {
  dot_product(poynting_flux, poynting_covector, normal_vector);
}

}  // namespace ForceFree
