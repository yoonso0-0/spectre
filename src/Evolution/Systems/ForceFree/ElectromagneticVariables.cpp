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

void electromagnetic_spatial_poynting_vector(
    const gsl::not_null<tnsr::I<DataVector, 3>*>
        electromagnetic_spatial_poynting_vector,
    const tnsr::I<DataVector, 3>& tilde_e,
    const tnsr::I<DataVector, 3>& tilde_b, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric) {
  const size_t num_grid_pts = get(sqrt_det_spatial_metric).size();

  Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::Tempi<0, 3>,
                       ::Tags::Tempi<1, 3>>>
      buffer{num_grid_pts};

  auto& energy_density_over_lapse = get<::Tags::TempScalar<0>>(buffer);
  auto& tilde_e_one_form = get<::Tags::Tempi<0, 3>>(buffer);
  auto& tilde_b_one_form = get<::Tags::Tempi<1, 3>>(buffer);

  electromagnetic_energy_density(make_not_null(&energy_density_over_lapse),
                                 tilde_e, tilde_b, sqrt_det_spatial_metric,
                                 spatial_metric);
  get(energy_density_over_lapse) /= get(lapse);

  set_number_of_grid_points(electromagnetic_spatial_poynting_vector,
                            num_grid_pts);

  for (size_t i = 0; i < 3; ++i) {
    (*electromagnetic_spatial_poynting_vector).get(i) =
        -get(energy_density_over_lapse) * shift.get(i);
  }

  for (LeviCivitaIterator<3> it; it; ++it) {
    const auto& i = it[0];
    const auto& j = it[1];
    const auto& k = it[2];
    (*electromagnetic_spatial_poynting_vector).get(i) +=
        it.sign() * tilde_e_one_form.get(j) * tilde_b_one_form.get(k) /
        pow<3>(get(sqrt_det_spatial_metric));
  }
}

void poynting_flux(
    const gsl::not_null<Scalar<DataVector>*> poynting_flux,
    const tnsr::I<DataVector, 3>& electromagnetic_spatial_poynting_vector,
    const tnsr::i<DataVector, 3>& normal_covector) {
  dot_product(poynting_flux, electromagnetic_spatial_poynting_vector,
              normal_covector);
}

}  // namespace ForceFree
