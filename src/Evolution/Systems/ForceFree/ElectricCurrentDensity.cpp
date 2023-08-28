// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/ElectricCurrentDensity.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/LeviCivitaIterator.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/Hydro/LorentzFactor.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

#include <iostream>

namespace ForceFree {

namespace {

template <bool IncludeExplicitPart, bool IncludeImplicitPart>
void tilde_j_impl(
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_j,
    const Scalar<DataVector>& tilde_q,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_e,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const double parallel_conductivity, const Scalar<DataVector>& lapse,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const std::optional<Scalar<DataVector>>& neutron_star_interior_mask) {
  static_assert(IncludeExplicitPart or IncludeImplicitPart);

  // FIXME : for subcell, handle this with Tags not destructive resize function.
  const size_t num_grid_pts = get(tilde_q).size();
  set_number_of_grid_points(tilde_j, num_grid_pts);

  Variables<tmpl::list<::Tags::Tempi<0, 3>, ::Tags::Tempi<1, 3>,
                       ::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
                       ::Tags::TempScalar<2>,
                       //
                       ::Tags::Tempi<2, 3>, ::Tags::TempScalar<3>,
                       ::Tags::TempScalar<4>, ::Tags::TempScalar<5>>>
      buffer{num_grid_pts};

  // compute one-forms of TildeE and TildeB in advance to reduce the number of
  // dot products using spatial metric (which are slower than dot products
  // without using spatial metric)
  auto& tilde_e_one_form = get<::Tags::Tempi<0, 3>>(buffer);
  auto& tilde_b_one_form = get<::Tags::Tempi<1, 3>>(buffer);
  raise_or_lower_index(make_not_null(&tilde_e_one_form), tilde_e,
                       spatial_metric);
  raise_or_lower_index(make_not_null(&tilde_b_one_form), tilde_b,
                       spatial_metric);

  // Compute \tilde{B}^2 = \tilde{B}^j \tilde{B}_j. We need this quantity for
  // both drift (explicit, non-stiff) and parallel (implicit, stiff) components
  // of J^i.
  auto& tilde_b_squared = get<::Tags::TempScalar<0>>(buffer);
  dot_product(make_not_null(&tilde_b_squared), tilde_b, tilde_b_one_form);

  if constexpr (IncludeImplicitPart) {
    (void)tilde_q;                  // avoid compiler warnings
    (void)sqrt_det_spatial_metric;  // avoid compiler warnings

    auto& tilde_e_squared = get<::Tags::TempScalar<1>>(buffer);
    auto& tilde_e_dot_tilde_b = get<::Tags::TempScalar<2>>(buffer);
    dot_product(make_not_null(&tilde_e_squared), tilde_e, tilde_e_one_form);
    dot_product(make_not_null(&tilde_e_dot_tilde_b), tilde_e, tilde_b_one_form);

    for (size_t i = 0; i < 3; ++i) {
      (*tilde_j).get(i) =
          parallel_conductivity *
          (get(tilde_e_dot_tilde_b) * tilde_b.get(i) +
           max(get(tilde_e_squared) - get(tilde_b_squared), 0.0) *
               tilde_e.get(i));
    }
  } else {
    // tilde_j should be initialized to zero before the summation performed in
    // the next `if constexpr` block
    for (size_t i = 0; i < 3; ++i) {
      (*tilde_j).get(i) = 0;
    }
  }

  if constexpr (IncludeExplicitPart) {
    (void)parallel_conductivity;  // avoid compiler warnings

    for (LeviCivitaIterator<3> it; it; ++it) {
      const auto& i = it[0];
      const auto& j = it[1];
      const auto& k = it[2];
      (*tilde_j).get(i) +=
          it.sign() * get(tilde_q) * tilde_e_one_form.get(j) *
          tilde_b_one_form.get(k) /
          get(sqrt_det_spatial_metric);  // the extra 1/sqrt{gamma} factor comes
                                         // from the spatial Levi-Civita tensor
    }
  }

  // overall factor
  for (size_t i = 0; i < 3; ++i) {
    (*tilde_j).get(i) *= get(lapse) / get(tilde_b_squared);
  }

  if (neutron_star_interior_mask.has_value()) {
    ASSERT(num_grid_pts == get(neutron_star_interior_mask.value()).size(),
           "Mask size (" << get(neutron_star_interior_mask.value()).size()
                         << ") does not match the number of grid points ("
                         << num_grid_pts << ") ");

    // std::cout << "Has Mask" << std::endl;

    for (size_t m = 0; m < num_grid_pts; ++m) {
      const auto& mask_value = get(*neutron_star_interior_mask)[m];

      if (mask_value > 0.0) {
        // Exterior (magnetosphere)
        continue;
      } else {
        // Interior (MHD condition)

        // When overwriting fields, just set to zero...
        for (size_t d = 0; d < 3; ++d) {
          (*tilde_j).get(d)[m] = 0.0;
        }

        // ------------------------ Resistive Ohm's law
        // Explicit part
        // if constexpr (IncludeExplicitPart) {
        //   for (size_t d = 0; d < 3; ++d) {
        //     // J^i = q v^i
        //     (*tilde_j).get(d)[m] = get(lapse)[m] * get(tilde_q)[m] *
        //                            ns_interior_spatial_velocity.get(d)[m];
        //   }
        // } else {
        //   for (size_t d = 0; d < 3; ++d) {
        //     (*tilde_j).get(d)[m] = 0.0;
        //   }
        // }
        // const double aws = get(lapse)[m] * get(ns_interior_lorentz_factor)[m]
        // *
        //                    parallel_conductivity;
        // if constexpr (IncludeImplicitPart) {
        //   for (LeviCivitaIterator<3> it; it; ++it) {
        //     const auto& i = it[0];
        //     const auto& j = it[1];
        //     const auto& k = it[2];
        //     (*tilde_j).get(i)[m] +=
        //         aws * it.sign() * ns_interior_spatial_velocity_form.get(j)[m]
        //         * tilde_b_one_form.get(k)[m] /
        //         get(sqrt_det_spatial_metric)[m];  // the extra 1/sqrt{gamma}
        //                                           // factor comes from the
        //                                           // spatial Levi-Civita
        //                                           tensor
        //   }

        //   for (size_t d = 0; d < 3; ++d) {
        //     (*tilde_j).get(d)[m] +=
        //         aws *
        //         (tilde_e.get(d)[m] - get(tilde_e_dot_ns_spatial_velocity)[m]
        //         *
        //                        ns_interior_spatial_velocity.get(d)[m]);
        //   }
        // }
      }
    }
  }
}

}  // namespace

void ComputeDriftTildeJ::apply(
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> drift_tilde_j,
    const Scalar<DataVector>& tilde_q,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_e,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const double parallel_conductivity, const Scalar<DataVector>& lapse,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const std::optional<Scalar<DataVector>>& neutron_star_interior_mask) {
  tilde_j_impl<true, false>(
      drift_tilde_j, tilde_q, tilde_e, tilde_b, parallel_conductivity, lapse,
      sqrt_det_spatial_metric, spatial_metric, neutron_star_interior_mask);
}

void ComputeParallelTildeJ::apply(
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        parallel_tilde_j,
    const Scalar<DataVector>& tilde_q,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_e,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const double parallel_conductivity, const Scalar<DataVector>& lapse,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const std::optional<Scalar<DataVector>>& neutron_star_interior_mask) {
  tilde_j_impl<false, true>(
      parallel_tilde_j, tilde_q, tilde_e, tilde_b, parallel_conductivity, lapse,
      sqrt_det_spatial_metric, spatial_metric, neutron_star_interior_mask);
}

namespace Tags {
void ComputeTildeJ::function(
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_j,
    const Scalar<DataVector>& tilde_q,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_e,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const double parallel_conductivity, const Scalar<DataVector>& lapse,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const std::optional<Scalar<DataVector>>& neutron_star_interior_mask) {
  tilde_j_impl<true, true>(
      tilde_j, tilde_q, tilde_e, tilde_b, parallel_conductivity, lapse,
      sqrt_det_spatial_metric, spatial_metric, neutron_star_interior_mask);
}
}  // namespace Tags

void StiffSourceTildeE ::apply(
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        stiff_source_tilde_e,
    const Scalar<DataVector>& tilde_q,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_e,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const double parallel_conductivity, const Scalar<DataVector>& lapse,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const std::optional<Scalar<DataVector>>& neutron_star_interior_mask) {
  ComputeParallelTildeJ::apply(stiff_source_tilde_e, tilde_q, tilde_e, tilde_b,
                               parallel_conductivity, lapse,
                               sqrt_det_spatial_metric, spatial_metric,
                               neutron_star_interior_mask);
  for (size_t i = 0; i < 3; ++i) {
    (*stiff_source_tilde_e).get(i) *= -1.0;
  }
  // std::cout << "Stiff source term : " << *stiff_source_tilde_e << std::endl;
}

void StiffSourceTildeEJacobian::apply(
    const gsl::not_null<tnsr::iJ<DataVector, 3>*> d_source_e_d_tilde_e,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_e,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const double parallel_conductivity, const Scalar<DataVector>& lapse,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const std::optional<Scalar<DataVector>>& neutron_star_interior_mask) {
  Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
                       ::Tags::Tempi<0, 3>, ::Tags::Tempi<1, 3>,
                       //
                       ::Tags::TempScalar<2>, ::Tags::Tempi<2, 3>>>
      buffer{get(lapse).size()};

  auto& tilde_e_squared = get<::Tags::TempScalar<0>>(buffer);
  auto& tilde_b_squared = get<::Tags::TempScalar<1>>(buffer);
  auto& tilde_e_one_form = get<::Tags::Tempi<0, 3>>(buffer);
  auto& tilde_b_one_form = get<::Tags::Tempi<1, 3>>(buffer);

  dot_product(make_not_null(&tilde_e_squared), tilde_e, tilde_e,
              spatial_metric);
  dot_product(make_not_null(&tilde_b_squared), tilde_b, tilde_b,
              spatial_metric);
  raise_or_lower_index(make_not_null(&tilde_e_one_form), tilde_e,
                       spatial_metric);
  raise_or_lower_index(make_not_null(&tilde_b_one_form), tilde_b,
                       spatial_metric);

  // note) ds_de.get(i,j) = ds^j / de^i
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      (*d_source_e_d_tilde_e).get(i, j) =
          tilde_b_one_form.get(i) * tilde_b.get(j) +
          2.0 * step_function(get(tilde_e_squared) - get(tilde_b_squared)) *
              tilde_e_one_form.get(i) * tilde_e.get(j);
    }

    (*d_source_e_d_tilde_e).get(i, i) +=
        max(get(tilde_e_squared) - get(tilde_b_squared), 0.0);

    for (size_t j = 0; j < 3; ++j) {
      (*d_source_e_d_tilde_e).get(i, j) *=
          -parallel_conductivity * get(lapse) / get(tilde_b_squared);
    }
  }

  if (neutron_star_interior_mask.has_value()) {
    for (size_t m = 0; m < get(lapse).size(); ++m) {
      const auto& mask = get(*neutron_star_interior_mask)[m];

      if (mask > 0.0) {  // Exterior (magnetosphere)
        // do nothing
        continue;
      } else {  // Interior (MHD condition)
        // When overwriting fields, just set to -1.0
        for (size_t i = 0; i < 3; ++i) {
          for (size_t j = 0; j < 3; ++j) {
            (*d_source_e_d_tilde_e).get(i, j)[m] = -1.0;
          }
        }
      }
    }
  }
}

}  // namespace ForceFree
