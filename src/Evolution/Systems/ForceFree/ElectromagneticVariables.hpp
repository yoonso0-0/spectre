// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Surfaces/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"

#include "NumericalAlgorithms/SphericalHarmonics/Tags.hpp"

#include <iostream>

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

/// ----------------------- Things related to observing Poynting flux

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
 * \brief Computes the Poynting vector with a lower index $S_i$.
 */
void poynting_covector(
    const gsl::not_null<tnsr::i<DataVector, 3>*> poynting_covector,
    const tnsr::I<DataVector, 3>& tilde_e,
    const tnsr::I<DataVector, 3>& tilde_b,
    const Scalar<DataVector>& sqrt_det_spatial_metric);

/*!
 * \brief Computes the Poynting flux dot normal vector $S_i n^i$.
 */
void poynting_flux(const gsl::not_null<Scalar<DataVector>*> poynting_flux,
                   const tnsr::i<DataVector, 3>& poynting_covector,
                   const tnsr::I<DataVector, 3>& normal_vector);

/*!
 * \brief Computes the magnetic field dot normal vector $B_i n^i$.
 */
void magnetic_flux(
    const gsl::not_null<Scalar<DataVector>*> magnetic_flux,
    const tnsr::I<DataVector, 3>& tilde_b,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::I<DataVector, 3, Frame::Inertial>& cartesian_coords,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_one_form);

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
struct PoyntingCovectorCompute : PoyntingCovector, db::ComputeTag {
  using argument_tags =
      tmpl::list<TildeE, TildeB, gr::Tags::SqrtDetSpatialMetric<DataVector>>;
  using return_type = tnsr::i<DataVector, 3>;
  using base = PoyntingCovector;

  static constexpr auto function = &poynting_covector;
};

struct PoyntingFluxCompute : PoyntingFlux, db::ComputeTag {
  using argument_tags =
      tmpl::list<PoyntingCovector,
                 ylm::Tags::UnitNormalVector<Frame::Inertial>>;
  using return_type = Scalar<DataVector>;
  using base = PoyntingFlux;

  static constexpr auto function = &poynting_flux;
};

struct MagneticFluxCompute : MagneticFlux, db::ComputeTag {
  using argument_tags =
      tmpl::list<TildeB, gr::Tags::SqrtDetSpatialMetric<DataVector>,
                 ylm::Tags::CartesianCoords<Frame::Inertial>,
                 ylm::Tags::UnitNormalOneForm<Frame::Inertial>>;
  using return_type = Scalar<DataVector>;
  using base = MagneticFlux;

  static constexpr auto function = &magnetic_flux;
};

}  // namespace Tags

}  // namespace ForceFree
