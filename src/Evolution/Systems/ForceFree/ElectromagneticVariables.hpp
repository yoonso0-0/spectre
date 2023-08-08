// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace ForceFree {

/*!
 * \brief Computes electric field $E^i$ from TildeE or magnetic field $B^i$ from
 * TildeB.
 */
void em_field_from_evolved_fields(
    const gsl::not_null<tnsr::I<DataVector, 3>*> vector,
    const tnsr::I<DataVector, 3>& densitized_vector,
    const Scalar<DataVector>& sqrt_det_spatial_metric);

/*!
 * \brief Computes electric charge density $q$ from TildeQ.
 */
void charge_density_from_tilde_q(
    const gsl::not_null<Scalar<DataVector>*> charge_density,
    const Scalar<DataVector>& tilde_q,
    const Scalar<DataVector>& sqrt_det_spatial_metric);

/*!
 * \brief Computes electric current density $J^i$ from TildeJ.
 */
void electric_current_density_from_tilde_j(
    const gsl::not_null<tnsr::I<DataVector, 3>*> electric_current_density,
    const tnsr::I<DataVector, 3>& tilde_j,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const Scalar<DataVector>& lapse);

/*!
 * \brief Computes energy dissipation rate $J_i E^i$.
 */
void joule_heating(const gsl::not_null<Scalar<DataVector>*> joule_heating,
                   const tnsr::I<DataVector, 3>& tilde_e,
                   const tnsr::I<DataVector, 3>& tilde_j,
                   const Scalar<DataVector>& lapse,
                   const Scalar<DataVector>& sqrt_det_spatial_metric);

namespace Tags {
/*!
 * \brief Compute item for electric field $E^i$ from TildeE.
 *
 * \note This ComputeTag is solely for observation purpose, not related to
 * actual time evolution.
 */
struct ElectricFieldCompute : ElectricField, db::ComputeTag {
  using argument_tags =
      tmpl::list<TildeE, gr::Tags::SqrtDetSpatialMetric<DataVector>>;
  using return_type = tnsr::I<DataVector, 3>;
  using base = ElectricField;

  static constexpr auto function = &em_field_from_evolved_fields;
};

/*!
 * \brief Compute item for magnetic field $B^i$ from TildeB.
 *
 * \note This ComputeTag is solely for observation purpose, not related to
 * actual time evolution.
 */
struct MagneticFieldCompute : MagneticField, db::ComputeTag {
  using argument_tags =
      tmpl::list<TildeB, gr::Tags::SqrtDetSpatialMetric<DataVector>>;
  using return_type = tnsr::I<DataVector, 3>;
  using base = MagneticField;

  static constexpr auto function = &em_field_from_evolved_fields;
};

/*!
 * \brief Compute item for electric charge density $q$ from TildeQ.
 *
 * \note This ComputeTag is solely for observation purpose, not related to
 * actual time evolution.
 */
struct ChargeDensityCompute : ChargeDensity, db::ComputeTag {
  using argument_tags =
      tmpl::list<TildeQ, gr::Tags::SqrtDetSpatialMetric<DataVector>>;
  using return_type = Scalar<DataVector>;
  using base = ChargeDensity;

  static constexpr auto function = &charge_density_from_tilde_q;
};

/*!
 * \brief Compute item for electric current density $J^i$ from TildeJ.
 *
 * \note This ComputeTag is solely for observation purpose, not related to
 * actual time evolution.
 */
struct ElectricCurrentDensityCompute : ElectricCurrentDensity, db::ComputeTag {
  using argument_tags = tmpl::list<TildeJ, gr::Tags::Lapse<DataVector>,
                                   gr::Tags::SqrtDetSpatialMetric<DataVector>>;
  using return_type = tnsr::I<DataVector, 3>;
  using base = ElectricCurrentDensity;

  static constexpr auto function = &electric_current_density_from_tilde_j;
};

/*!
 * \brief Compute item for energy dissipation $J^i E_i$.
 *
 */
struct JouleHeatingCompute : JouleHeating, db::ComputeTag {
  using argument_tags = tmpl::list<TildeE, TildeJ, gr::Tags::Lapse<DataVector>,
                                   gr::Tags::SqrtDetSpatialMetric<DataVector>>;
  using return_type = Scalar<DataVector>;
  using base = JouleHeating;

  static constexpr auto function = &joule_heating;
};
}  // namespace Tags

///

/*!
 * \brief Computes the electromagnetic energy density.
 */
void electromagnetic_energy_density(
    const gsl::not_null<Scalar<DataVector>*> electromagnetic_energy_density,
    const tnsr::I<DataVector, 3>& tilde_e,
    const tnsr::I<DataVector, 3>& tilde_b,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric);

/*!
 * \brief Computes the Poynting vector
 */
void electromagnetic_spatial_poynting_vector(
    const gsl::not_null<tnsr::I<DataVector, 3>*>
        electromagnetic_spatial_poynting_vector,
    const tnsr::I<DataVector, 3>& tilde_e,
    const tnsr::I<DataVector, 3>& tilde_b, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric);

/*!
 * \brief Computes the Poynting flux
 */
void poynting_flux(
    const gsl::not_null<Scalar<DataVector>*> poynting_flux,
    const tnsr::I<DataVector, 3>& electromagnetic_spatial_poynting_vector,
    const tnsr::i<DataVector, 3>& normal_covector);

namespace Tags {
/*!
 * \brief Computes the electromagnetic energy denisty.
 *
 */
struct ElectromagneticEnergyDensityCompute : ElectromagneticEnergyDensity,
                                             db::ComputeTag {
  using argument_tags =
      tmpl::list<TildeE, TildeB, gr::Tags::SqrtDetSpatialMetric<DataVector>,
                 gr::Tags::SpatialMetric<DataVector, 3>>;
  using return_type = Scalar<DataVector>;
  using base = ElectromagneticEnergyDensity;

  static constexpr auto function = &electromagnetic_energy_density;
};

/*!
 * \brief Computes the electromagnetic Poynting vector $S^i$.
 *
 */
struct ElectromagneticSpatialPoyntingVectorCompute
    : ElectromagneticSpatialPoyntingVector,
      db::ComputeTag {
  using argument_tags = tmpl::list<TildeE, TildeB, gr::Tags::Lapse<DataVector>,
                                   gr::Tags::Shift<DataVector, 3>,
                                   gr::Tags::SqrtDetSpatialMetric<DataVector>,
                                   gr::Tags::SpatialMetric<DataVector, 3>>;
  using return_type = tnsr::I<DataVector, 3>;
  using base = ElectromagneticSpatialPoyntingVector;

  static constexpr auto function = &electromagnetic_spatial_poynting_vector;
};

}  // namespace Tags

}  // namespace ForceFree
