// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Tags.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/Sphere.hpp"
#include "PointwiseFunctions/GeneralRelativity/DetAndInverseSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Surfaces/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags/Time.hpp"

namespace grmhd::ValenciaDivClean {

//
//
void momentum_outflux_covector(
    gsl::not_null<tnsr::i<DataVector, 3>*> momentum_outflux_covector,
    const Scalar<DataVector>& rest_mass_density
    //
    // ...
    //
);

void momentum_outflux(gsl::not_null<Scalar<DataVector>*> momentum_outflux,
                      const Scalar<DataVector>& rest_mass_density
                      //
                      // ...
                      //
);

//
//
void gravitational_drag_source_term(
    gsl::not_null<tnsr::i<DataVector, 3>*> result,
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
    const tnsr::i<DataVector, 3>& d_lapse,
    const tnsr::iJ<DataVector, 3>& d_shift,
    const tnsr::ijj<DataVector, 3>& d_spatial_metric);

//
//
//
struct MomentumOutfluxThroughSphere
    : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
  using temporal_id = ::Tags::Time;

  // quantities required to compute the surface integral
  using vars_to_interpolate_to_target = tmpl::list<>;

  using compute_items_on_source = tmpl::list<
      //
      gr::Tags::DetAndInverseSpatialMetricCompute<DataVector, 3,
                                                  Frame::Inertial>,
      gr::Tags::SqrtDetSpatialMetricCompute<DataVector, 3, Frame::Inertial>,
      ylm::Tags::OneOverOneFormMagnitudeCompute<DataVector, 3, Frame::Inertial>,
      ylm::Tags::UnitNormalOneFormCompute<Frame::Inertial>,
      ylm::Tags::UnitNormalVectorCompute<Frame::Inertial>,
      //
      gr::surfaces::Tags::AreaElementCompute<Frame::Inertial>,
      gr::surfaces::Tags::SurfaceIntegralCompute<
          ValenciaDivClean::Tags::MomentumOutflux, Frame::Inertial>
      //
      >;

  using compute_items_on_target = tmpl::list<>;

  using compute_target_points =
      intrp::TargetPoints::Sphere<MomentumOutfluxThroughSphere,
                                  ::Frame::Inertial>;

  using post_interpolation_callbacks = tmpl::list<>;

  template <typename Metavariables>
  using interpolating_component =
      typename Metavariables::dg_element_array_component;
};

// --------------------------------------------------------------------------

//
//
//
struct GravitationalDragSourceTerm
    : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
  using temporal_id = ::Tags::Time;

  // quantities required to compute the surface integral
  using vars_to_interpolate_to_target = tmpl::list<>;

  using compute_items_on_source = tmpl::list<>;

  using compute_items_on_target = tmpl::list<>;

  using compute_target_points =
      intrp::TargetPoints::Sphere<GravitationalDragSourceTerm,
                                  ::Frame::Inertial>;

  using post_interpolation_callbacks = tmpl::list<>;

  template <typename Metavariables>
  using interpolating_component =
      typename Metavariables::dg_element_array_component;
};

}  // namespace grmhd::ValenciaDivClean
