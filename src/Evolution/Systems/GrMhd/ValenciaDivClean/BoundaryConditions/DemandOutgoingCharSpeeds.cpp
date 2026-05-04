// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/DemandOutgoingCharSpeeds.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/SliceTensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ComputeFluxesFromPrimitives.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean::BoundaryConditions {
DemandOutgoingCharSpeeds::DemandOutgoingCharSpeeds(CkMigrateMessage* const msg)
    : BoundaryCondition(msg) {}

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
DemandOutgoingCharSpeeds::get_clone() const {
  return std::make_unique<DemandOutgoingCharSpeeds>(*this);
}

void DemandOutgoingCharSpeeds::pup(PUP::er& p) { BoundaryCondition::pup(p); }

std::optional<std::string>
DemandOutgoingCharSpeeds::dg_demand_outgoing_char_speeds(
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
        face_mesh_velocity,
    const tnsr::i<DataVector, 3, Frame::Inertial>&
        outward_directed_normal_covector,
    const tnsr::I<DataVector, 3, Frame::Inertial>&
    /*outward_directed_normal_vector*/,

    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const Scalar<DataVector>& lapse) {
  double min_speed = std::numeric_limits<double>::signaling_NaN();
  Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>>> buffer{
      get(lapse).size()};
  auto& normal_dot_shift = get<::Tags::TempScalar<0>>(buffer);
  dot_product(make_not_null(&normal_dot_shift),
              outward_directed_normal_covector, shift);
  if (face_mesh_velocity.has_value()) {
    auto& normal_dot_mesh_velocity = get<::Tags::TempScalar<1>>(buffer);
    dot_product(make_not_null(&normal_dot_mesh_velocity),
                outward_directed_normal_covector, face_mesh_velocity.value());
    get(normal_dot_shift) += get(normal_dot_mesh_velocity);
  }
  // The characteristic speeds are bounded by \pm \alpha - \beta_n, and
  // saturate that bound, so there is no need to check the hydro-dependent
  // characteristic speeds.
  min_speed = std::min(min(-get(lapse) - get(normal_dot_shift)),
                       min(get(lapse) - get(normal_dot_shift)));
  if (min_speed < 0.0) {
    return {MakeString{}
            << "DemandOutgoingCharSpeeds boundary condition violated. Speed: "
            << min_speed << "\nn_i: " << outward_directed_normal_covector
            << "\n"};
  }
  return std::nullopt;
}

void DemandOutgoingCharSpeeds::fd_demand_outgoing_char_speeds(
    const gsl::not_null<Scalar<DataVector>*> rest_mass_density,
    const gsl::not_null<Scalar<DataVector>*> electron_fraction,
    const gsl::not_null<Scalar<DataVector>*> pressure,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        lorentz_factor_times_spatial_velocity,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        magnetic_field,
    const gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field,

    const gsl::not_null<std::optional<Variables<db::wrap_tags_in<
        Flux, typename grmhd::ValenciaDivClean::System::flux_variables>>>*>
        cell_centered_ghost_fluxes,

    const Direction<3>& direction,

    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
        dg_volume_mesh_velocity,

    // fd_interior_temporary_tags
    const Scalar<DataVector>& interior_lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_shift,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& interior_spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& interior_inv_spatial_metric,
    const ::InverseJacobian<DataVector, 3, Frame::ElementLogical,
                            Frame::Inertial>& inv_jacobian_dg,
    const Mesh<3>& dg_mesh, const Mesh<3>& subcell_mesh,

    // fd_interior_primitive_variables_tags
    const Scalar<DataVector>& interior_rest_mass_density,
    const Scalar<DataVector>& interior_electron_fraction,
    const Scalar<DataVector>& interior_pressure,
    const Scalar<DataVector>& interior_specific_internal_energy,
    const Scalar<DataVector>& interior_lorentz_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_spatial_velocity,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_magnetic_field,
    const Scalar<DataVector>& interior_divergence_cleaning_field,

    // fd_gridless_tags
    const fd::Reconstructor& reconstructor) {
  double min_char_speed = std::numeric_limits<double>::signaling_NaN();

  const size_t ghost_zone_size{reconstructor.ghost_zone_size()};
  const size_t dim_direction{direction.dimension()};

  const auto subcell_extents{subcell_mesh.extents()};
  const size_t num_subcell_face_pts{
      subcell_extents.slice_away(dim_direction).product()};

  // The boundary condition below simply uses the outermost values on
  // cell-centered FD grid points to compute face values on the external
  // boundary. This is equivalent to adopting the piecewise constant (lowest
  // order) FD reconstruction for FD cells at the external boundaries.
  //
  // In the future we may want to use more accurate methods (for instance,
  // one-sided characteristic reconstruction using WENO) for imposing
  // higher-order DemandOutgoingCharSpeeds boundary condition.

  // We need three Variables objects to store:
  //  1. Reconstructed variables at the innermost ghost zone (a single slice)
  //  2. Reconstructed variables at the all ghost zone (multiple slices with
  //     depth being `ghost_zone_size`)
  //  3. Temporary quantities (e.g. subcell normal vector, normal_dot_shift ..)
  //
  // So allocate a giant buffer to store all of these

  Variables<tmpl::list<
      // Metric quantities
      gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
      gr::Tags::InverseSpatialMetric<DataVector, 3>,
      // Temporaries
      ::Tags::TempScalar<0>, ::Tags::Tempi<0, 3>, ::Tags::TempI<0, 3>>>
      temporary_vars{num_subcell_face_pts, 0.0};

  auto& lapse_at_boundary = get<gr::Tags::Lapse<DataVector>>(temporary_vars);
  auto& shift_at_boundary = get<gr::Tags::Shift<DataVector, 3>>(temporary_vars);
  auto& inv_spatial_metric_at_boundary =
      get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(temporary_vars);
  evolution::dg::subcell::slice_tensor_for_subcell(
      make_not_null(&lapse_at_boundary), interior_lapse, subcell_extents, 1,
      direction, {});
  evolution::dg::subcell::slice_tensor_for_subcell(
      make_not_null(&shift_at_boundary), interior_shift, subcell_extents, 1,
      direction, {});
  evolution::dg::subcell::slice_tensor_for_subcell(
      make_not_null(&inv_spatial_metric_at_boundary),
      interior_inv_spatial_metric, subcell_extents, 1, direction, {});

  // Construct normal vector and normal covector on the subcell face mesh.
  // Project normal covector from DG to FD first, then compute normal vector on
  // FD mesh using inverse spatial metric
  auto& subcell_face_normal_covector = get<::Tags::Tempi<0, 3>>(temporary_vars);
  {
    // The unnormalized normal vector is n_j = d \xi^{\hat i}/dx^j
    // with "i" the current face.
    tnsr::i<DataVector, 3, Frame::Inertial> subcell_volume_normal_covector{
        subcell_extents.product(), 0.0};
    for (size_t j = 0; j < 3; ++j) {
      subcell_volume_normal_covector.get(j) =
          evolution::dg::subcell::fd::project_to_faces(
              inv_jacobian_dg.get(dim_direction, j), dg_mesh, subcell_extents,
              dim_direction);
    }
    evolution::dg::subcell::slice_tensor_for_subcell(
        make_not_null(&subcell_face_normal_covector),
        subcell_volume_normal_covector, subcell_extents, 1, direction, {});

    // Apply normalization to covector and compute normal vector n^j
    const Scalar<DataVector> normalization{sqrt(get(
        dot_product(subcell_face_normal_covector, subcell_face_normal_covector,
                    inv_spatial_metric_at_boundary)))};
    for (size_t j = 0; j < 3; j++) {
      subcell_face_normal_covector.get(j) *=
          direction.sign() / get(normalization);
    }
  }

  auto& normal_dot_shift = get<::Tags::TempScalar<0>>(temporary_vars);
  dot_product(make_not_null(&normal_dot_shift), subcell_face_normal_covector,
              shift_at_boundary);

  if (dg_volume_mesh_velocity.has_value()) {
    // We need to compute subcell mesh velocity at the boundary.
    // Project the DG mesh velocity onto subcell grid, then slice the outermost
    // one.
    tnsr::I<DataVector, 3, Frame::Inertial> subcell_volume_mesh_velocity{
        subcell_extents.product(), 0.0};
    for (size_t j = 0; j < 3; ++j) {
      subcell_volume_mesh_velocity.get(j) =
          evolution::dg::subcell::fd::project_to_faces(
              inv_jacobian_dg.get(dim_direction, j), dg_mesh, subcell_extents,
              dim_direction);
    }

    auto& subcell_face_mesh_velocity = get<::Tags::TempI<0, 3>>(temporary_vars);
    evolution::dg::subcell::slice_tensor_for_subcell(
        make_not_null(&subcell_face_mesh_velocity),
        subcell_volume_mesh_velocity, subcell_extents, 1, direction, {});

    auto& normal_dot_mesh_velocity = get<::Tags::TempScalar<0>>(temporary_vars);
    dot_product(make_not_null(&normal_dot_mesh_velocity),
                subcell_face_normal_covector, subcell_face_mesh_velocity);
    get(normal_dot_shift) += get(normal_dot_mesh_velocity);
  }

  // The characteristic speeds are bounded by \pm \alpha - \beta_n, and
  // saturate that bound, so there is no need to check the hydro-dependent
  // characteristic speeds.
  min_char_speed =
      std::min(min(-get(lapse_at_boundary) - get(normal_dot_shift)),
               min(get(lapse_at_boundary) - get(normal_dot_shift)));

  if (min_char_speed < 0.0) {
    ERROR(
        "Subcell DemandOutgoingCharSpeeds boundary condition violated. Speed: "
        << min_char_speed << "\nn_i: " << subcell_face_normal_covector << "\n");
  } else {
    // Once the DemandOutgoingCharSpeeds condition has been checked, we fill
    // each slices of the ghost data with the boundary values. The reason that
    // we need this step is to prevent floating point exceptions being raised
    // while computing the subcell time derivative because of NaN or
    // uninitialized values in ghost data.

    using RestMassDensity = hydro::Tags::RestMassDensity<DataVector>;
    using ElectronFraction = hydro::Tags::ElectronFraction<DataVector>;
    using Pressure = hydro::Tags::Pressure<DataVector>;
    using SpecificInternalEnergy =
        hydro::Tags::SpecificInternalEnergy<DataVector>;
    using LorentzFactorTimesSpatialVelocity =
        hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>;
    using MagneticField = hydro::Tags::MagneticField<DataVector, 3>;
    using DivergenceCleaningField =
        hydro::Tags::DivergenceCleaningField<DataVector>;
    using SpatialVelocity = hydro::Tags::SpatialVelocity<DataVector, 3>;
    using LorentzFactor = hydro::Tags::LorentzFactor<DataVector>;
    using SqrtDetSpatialMetric = gr::Tags::SqrtDetSpatialMetric<DataVector>;
    using SpatialMetric = gr::Tags::SpatialMetric<DataVector, 3>;
    using InvSpatialMetric = gr::Tags::InverseSpatialMetric<DataVector, 3>;
    using Lapse = gr::Tags::Lapse<DataVector>;
    using Shift = gr::Tags::Shift<DataVector, 3>;

    using prim_tags_for_reconstruction =
        tmpl::list<RestMassDensity, ElectronFraction, Pressure,
                   LorentzFactorTimesSpatialVelocity, MagneticField,
                   DivergenceCleaningField>;
    using fluxes_tags =
        tmpl::push_back<typename System::primitive_variables_tag::tags_list,
                        LorentzFactorTimesSpatialVelocity, SpatialMetric, Lapse,
                        Shift, SqrtDetSpatialMetric, InvSpatialMetric>;

    const bool need_tags_for_fluxes = cell_centered_ghost_fluxes->has_value();
    // Create a single large DV to reduce the number of Variables allocations
    const size_t buffer_size_for_fluxes =
        need_tags_for_fluxes
            ? Variables<fluxes_tags>::number_of_independent_components
            : 0;
    const size_t buffer_size_per_grid_pts = Variables<
        prim_tags_for_reconstruction>::number_of_independent_components;
    DataVector buffer_for_boundary_and_ghost_vars{
        (buffer_size_per_grid_pts + buffer_size_for_fluxes) *
            num_subcell_face_pts * (1 + ghost_zone_size),
        0.0};

    // a Variables object to store prim variables on outermost layer of FD grid
    // points
    Variables<prim_tags_for_reconstruction> boundary_vars{
        buffer_for_boundary_and_ghost_vars.data(),
        buffer_size_per_grid_pts * num_subcell_face_pts};
    // a Variables object to store prim variables on ghost zone
    Variables<prim_tags_for_reconstruction> ghost_vars{
        std::next(buffer_for_boundary_and_ghost_vars.data(),
                  static_cast<std::ptrdiff_t>(boundary_vars.size())),
        buffer_size_per_grid_pts * num_subcell_face_pts * ghost_zone_size};

    auto get_boundary_val = [&direction, &subcell_extents](auto volume_tensor) {
      return evolution::dg::subcell::slice_tensor_for_subcell(
          volume_tensor, subcell_extents, 1, direction, {});
    };

    get<RestMassDensity>(boundary_vars) =
        get_boundary_val(interior_rest_mass_density);
    get<ElectronFraction>(boundary_vars) =
        get_boundary_val(interior_electron_fraction);
    get<Pressure>(boundary_vars) = get_boundary_val(interior_pressure);
    // Note : 'lorentz factor times spatial velocity' needs to be in the FD
    // ghost data for reconstruction, instead of lorentz factor and spatial
    // velocity separately.
    for (size_t i = 0; i < 3; ++i) {
      get<LorentzFactorTimesSpatialVelocity>(boundary_vars).get(i) =
          get(get_boundary_val(interior_lorentz_factor)) *
          get_boundary_val(interior_spatial_velocity).get(i);
    }
    get<MagneticField>(boundary_vars) =
        get_boundary_val(interior_magnetic_field);
    get<DivergenceCleaningField>(boundary_vars) =
        get_boundary_val(interior_divergence_cleaning_field);

    // Next, copy boundary values into each slices of ghost zone vars
    Index<3> ghost_data_extents = subcell_extents;
    ghost_data_extents[dim_direction] = ghost_zone_size;

    for (size_t i_ghost = 0; i_ghost < ghost_zone_size; ++i_ghost) {
      add_slice_to_data(make_not_null(&ghost_vars), boundary_vars,
                        ghost_data_extents, dim_direction, i_ghost);
    }

    *rest_mass_density = get<RestMassDensity>(ghost_vars);
    *electron_fraction = get<ElectronFraction>(ghost_vars);
    *pressure = get<Pressure>(ghost_vars);
    *lorentz_factor_times_spatial_velocity =
        get<LorentzFactorTimesSpatialVelocity>(ghost_vars);
    *magnetic_field = get<MagneticField>(ghost_vars);
    *divergence_cleaning_field = get<DivergenceCleaningField>(ghost_vars);

    if (need_tags_for_fluxes) {
      Variables<fluxes_tags> outermost_fluxes_vars{
          std::next(ghost_vars.data(),
                    static_cast<std::ptrdiff_t>(ghost_vars.size())),
          num_subcell_face_pts * buffer_size_for_fluxes};
      Variables<fluxes_tags> ghost_fluxes_vars{
          std::next(outermost_fluxes_vars.data(),
                    static_cast<std::ptrdiff_t>(outermost_fluxes_vars.size())),
          num_subcell_face_pts * buffer_size_for_fluxes * ghost_zone_size};

      get<RestMassDensity>(ghost_fluxes_vars) = *rest_mass_density;
      get<ElectronFraction>(ghost_fluxes_vars) = *electron_fraction;
      get<Pressure>(ghost_fluxes_vars) = *pressure;
      get<MagneticField>(ghost_fluxes_vars) = *magnetic_field;
      get<DivergenceCleaningField>(ghost_fluxes_vars) =
          *divergence_cleaning_field;

      get<SpecificInternalEnergy>(outermost_fluxes_vars) =
          get_boundary_val(interior_specific_internal_energy);
      get<SpatialMetric>(outermost_fluxes_vars) =
          get_boundary_val(interior_spatial_metric);
      get<Lapse>(outermost_fluxes_vars) = get_boundary_val(interior_lapse);
      get<Shift>(outermost_fluxes_vars) = get_boundary_val(interior_shift);

      for (size_t i_ghost = 0; i_ghost < ghost_zone_size; ++i_ghost) {
        add_slice_to_data(make_not_null(&ghost_fluxes_vars),
                          outermost_fluxes_vars, ghost_data_extents,
                          dim_direction, i_ghost);
      }

      determinant_and_inverse(
          make_not_null(&get<SqrtDetSpatialMetric>(ghost_fluxes_vars)),
          make_not_null(&get<InvSpatialMetric>(ghost_fluxes_vars)),
          get<SpatialMetric>(ghost_fluxes_vars));
      get(get<SqrtDetSpatialMetric>(ghost_fluxes_vars)) =
          sqrt(get(get<SqrtDetSpatialMetric>(ghost_fluxes_vars)));
      tenex::evaluate(
          make_not_null(&get<LorentzFactor>(ghost_fluxes_vars)),
          sqrt(1.0 + get<SpatialMetric>(ghost_fluxes_vars)(ti::i, ti::j) *
                         (*lorentz_factor_times_spatial_velocity)(ti::I) *
                         (*lorentz_factor_times_spatial_velocity)(ti::J)));
      tenex::evaluate<ti::I>(
          make_not_null(&get<SpatialVelocity>(ghost_fluxes_vars)),
          (*lorentz_factor_times_spatial_velocity)(ti::I) /
              get<LorentzFactor>(ghost_fluxes_vars)());

      Variables<typename System::variables_tag::tags_list> temp_vars(
          get(*rest_mass_density).size());

      compute_fluxes_from_primitives(
          make_not_null(&cell_centered_ghost_fluxes->value()),
          ghost_fluxes_vars);
    }
  }
}

// NOLINTNEXTLINE
PUP::able::PUP_ID DemandOutgoingCharSpeeds::my_PUP_ID = 0;

}  // namespace grmhd::ValenciaDivClean::BoundaryConditions
