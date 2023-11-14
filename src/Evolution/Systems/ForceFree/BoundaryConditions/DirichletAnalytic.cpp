// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/BoundaryConditions/DirichletAnalytic.hpp"

#include <memory>
#include <pup.h>
#include <utility>

#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/ElementMap.hpp"
#include "Evolution/DgSubcell/GhostZoneLogicalCoordinates.hpp"
#include "Evolution/Systems/ForceFree/ElectricCurrentDensity.hpp"
#include "Evolution/Systems/ForceFree/Fluxes.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "PointwiseFunctions/AnalyticData/ForceFree/Factory.hpp"
#include "PointwiseFunctions/AnalyticSolutions/ForceFree/Factory.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TaggedTuple.hpp"

#include <iostream>

namespace ForceFree::BoundaryConditions {

DirichletAnalytic::DirichletAnalytic(const DirichletAnalytic& rhs)
    : BoundaryCondition{dynamic_cast<const BoundaryCondition&>(rhs)},
      analytic_prescription_(rhs.analytic_prescription_->get_clone()) {}

DirichletAnalytic& DirichletAnalytic::operator=(const DirichletAnalytic& rhs) {
  if (&rhs == this) {
    return *this;
  }
  analytic_prescription_ = rhs.analytic_prescription_->get_clone();
  return *this;
}

DirichletAnalytic::DirichletAnalytic(
    std::unique_ptr<evolution::initial_data::InitialData> analytic_prescription)
    : analytic_prescription_(std::move(analytic_prescription)) {}

DirichletAnalytic::DirichletAnalytic(CkMigrateMessage* const msg)
    : BoundaryCondition(msg) {}

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
DirichletAnalytic::get_clone() const {
  return std::make_unique<DirichletAnalytic>(*this);
}

void DirichletAnalytic::pup(PUP::er& p) {
  BoundaryCondition::pup(p);
  p | analytic_prescription_;
}

// NOLINTNEXTLINE
PUP::able::PUP_ID DirichletAnalytic::my_PUP_ID = 0;

std::optional<std::string> DirichletAnalytic::dg_ghost(
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_e,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_b,
    const gsl::not_null<Scalar<DataVector>*> tilde_psi,
    const gsl::not_null<Scalar<DataVector>*> tilde_phi,
    const gsl::not_null<Scalar<DataVector>*> tilde_q,

    const gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*> tilde_e_flux,
    const gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*> tilde_b_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        tilde_psi_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        tilde_phi_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_q_flux,

    const gsl::not_null<Scalar<DataVector>*> lapse,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> shift,
    const gsl::not_null<tnsr::II<DataVector, 3, Frame::Inertial>*>
        inv_spatial_metric,

    const std::optional<
        tnsr::I<DataVector, 3, Frame::Inertial>>& /*face_mesh_velocity*/,
    const tnsr::i<DataVector, 3, Frame::Inertial>& /*normal_covector*/,
    const tnsr::I<DataVector, 3, Frame::Inertial>& /*normal_vector*/,

    const tnsr::I<DataVector, 3, Frame::Inertial>& coords,

    [[maybe_unused]] const double time, const double parallel_conductivity,
    const std::optional<Scalar<DataVector>>& neutron_star_interior_mask) const {
  auto boundary_values = call_with_dynamic_type<
      tuples::TaggedTuple<Tags::TildeE, Tags::TildeB, Tags::TildePsi,
                          Tags::TildePhi, Tags::TildeQ,
                          gr::Tags::Lapse<DataVector>,
                          gr::Tags::Shift<DataVector, 3>,
                          gr::Tags::SqrtDetSpatialMetric<DataVector>,
                          gr::Tags::SpatialMetric<DataVector, 3>,
                          gr::Tags::InverseSpatialMetric<DataVector, 3>>,
      tmpl::append<ForceFree::Solutions::all_solutions,
                   ForceFree::AnalyticData::all_data>>(
      analytic_prescription_.get(),
      [&coords, &time](const auto* const analytic_solution_or_data) {
        if constexpr (is_analytic_solution_v<
                          std::decay_t<decltype(*analytic_solution_or_data)>>) {
          return analytic_solution_or_data->variables(
              coords, time,
              tmpl::list<Tags::TildeE, Tags::TildeB, Tags::TildePsi,
                         Tags::TildePhi, Tags::TildeQ,
                         gr::Tags::Lapse<DataVector>,
                         gr::Tags::Shift<DataVector, 3>,
                         gr::Tags::SqrtDetSpatialMetric<DataVector>,
                         gr::Tags::SpatialMetric<DataVector, 3>,
                         gr::Tags::InverseSpatialMetric<DataVector, 3>>{});
        } else {
          (void)time;
          return analytic_solution_or_data->variables(
              coords,
              tmpl::list<Tags::TildeE, Tags::TildeB, Tags::TildePsi,
                         Tags::TildePhi, Tags::TildeQ,
                         gr::Tags::Lapse<DataVector>,
                         gr::Tags::Shift<DataVector, 3>,
                         gr::Tags::SqrtDetSpatialMetric<DataVector>,
                         gr::Tags::SpatialMetric<DataVector, 3>,
                         gr::Tags::InverseSpatialMetric<DataVector, 3>>{});
        }
      });

  *tilde_e = get<Tags::TildeE>(boundary_values);
  *tilde_b = get<Tags::TildeB>(boundary_values);
  *tilde_psi = get<Tags::TildePsi>(boundary_values);
  *tilde_phi = get<Tags::TildePhi>(boundary_values);
  *tilde_q = get<Tags::TildeQ>(boundary_values);

  *lapse = get<gr::Tags::Lapse<DataVector>>(boundary_values);
  *shift = get<gr::Tags::Shift<DataVector, 3>>(boundary_values);
  *inv_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(boundary_values);

  const auto& sqrt_det_spatial_metric =
      get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(boundary_values);
  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<DataVector, 3>>(boundary_values);

  if (neutron_star_interior_mask.has_value()) {
    ERROR(
        "Masked interior region should not cross the external boundary of "
        "domain.");
  }

  // allocate a temp buffer to compute \tilde{J}^i
  Variables<tmpl::list<::Tags::TempI<0, 3>>> buffer{get(*tilde_q).size()};
  auto& tilde_j = get<::Tags::TempI<0, 3>>(buffer);
  Tags::ComputeTildeJ::function(make_not_null(&tilde_j), *tilde_q, *tilde_e,
                                *tilde_b, parallel_conductivity, *lapse,
                                sqrt_det_spatial_metric, spatial_metric,
                                std::optional<Scalar<DataVector>>{});

  // compute corresponding fluxes
  ForceFree::Fluxes::apply(
      tilde_e_flux, tilde_b_flux, tilde_psi_flux, tilde_phi_flux, tilde_q_flux,
      *tilde_e, *tilde_b, *tilde_psi, *tilde_phi, *tilde_q, tilde_j, *lapse,
      *shift, sqrt_det_spatial_metric, spatial_metric, *inv_spatial_metric);

  return std::nullopt;
}

void DirichletAnalytic::fd_ghost(
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_j,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_e,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_b,
    const gsl::not_null<Scalar<DataVector>*> tilde_psi,
    const gsl::not_null<Scalar<DataVector>*> tilde_phi,
    const gsl::not_null<Scalar<DataVector>*> tilde_q,

    const Direction<3>& direction,

    // fd_interior_temporary_tags
    const Mesh<3> subcell_mesh,

    // fd_gridless_tags
    const double parallel_conductivity,
    const std::optional<Scalar<DataVector>> neutron_star_interior_mask,
    const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time,
    const ElementMap<3, Frame::Grid>& logical_to_grid_map,
    const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>&
        grid_to_inertial_map,
    const fd::Reconstructor& reconstructor) const {
  const size_t ghost_zone_size{reconstructor.ghost_zone_size()};

  const auto ghost_logical_coords =
      evolution::dg::subcell::fd::ghost_zone_logical_coordinates(
          subcell_mesh, ghost_zone_size, direction);

  const auto ghost_inertial_coords = grid_to_inertial_map(
      logical_to_grid_map(ghost_logical_coords), time, functions_of_time);

  // Compute the FD ghost data
  auto fd_ghost_values = call_with_dynamic_type<
      tuples::TaggedTuple<Tags::TildeE, Tags::TildeB, Tags::TildePsi,
                          Tags::TildePhi, Tags::TildeQ,
                          gr::Tags::Lapse<DataVector>,
                          gr::Tags::SqrtDetSpatialMetric<DataVector>,
                          gr::Tags::SpatialMetric<DataVector, 3>>,
      tmpl::append<ForceFree::Solutions::all_solutions,
                   ForceFree::AnalyticData::all_data>>(
      analytic_prescription_.get(),
      [&ghost_inertial_coords,
       &time](const auto* const analytic_solution_or_data) {
        if constexpr (is_analytic_solution_v<
                          std::decay_t<decltype(*analytic_solution_or_data)>>) {
          return analytic_solution_or_data->variables(
              ghost_inertial_coords, time,
              tmpl::list<Tags::TildeE, Tags::TildeB, Tags::TildePsi,
                         Tags::TildePhi, Tags::TildeQ,
                         gr::Tags::Lapse<DataVector>,
                         gr::Tags::SqrtDetSpatialMetric<DataVector>,
                         gr::Tags::SpatialMetric<DataVector, 3>>{});
        } else {
          (void)time;
          return analytic_solution_or_data->variables(
              ghost_inertial_coords,
              tmpl::list<Tags::TildeE, Tags::TildeB, Tags::TildePsi,
                         Tags::TildePhi, Tags::TildeQ,
                         gr::Tags::Lapse<DataVector>,
                         gr::Tags::SqrtDetSpatialMetric<DataVector>,
                         gr::Tags::SpatialMetric<DataVector, 3>>{});
        }
      });

  for (size_t i = 0; i < 3; ++i) {
    (*tilde_e).get(i) = get<Tags::TildeE>(fd_ghost_values).get(i);
    (*tilde_b).get(i) = get<Tags::TildeB>(fd_ghost_values).get(i);
  }
  get(*tilde_psi) = get(get<Tags::TildePsi>(fd_ghost_values));
  get(*tilde_phi) = get(get<Tags::TildePhi>(fd_ghost_values));
  get(*tilde_q) = get(get<Tags::TildeQ>(fd_ghost_values));

  if (neutron_star_interior_mask.has_value()) {
    ERROR(
        "Masked interior region should not cross the external boundary of "
        "domain.");
  }

  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(fd_ghost_values);
  const auto& sqrt_det_spatial_metric =
      get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(fd_ghost_values);
  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<DataVector, 3>>(fd_ghost_values);

  Tags::ComputeTildeJ::function(tilde_j, *tilde_q, *tilde_e, *tilde_b,
                                parallel_conductivity, lapse,
                                sqrt_det_spatial_metric, spatial_metric,
                                std::optional<Scalar<DataVector>>{});
}

}  // namespace ForceFree::BoundaryConditions
