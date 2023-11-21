// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <type_traits>
#include <vector>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/Factory3D.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Actions/RunEventsAndDenseTriggers.hpp"
#include "Evolution/Actions/RunEventsAndTriggers.hpp"
#include "Evolution/ComputeTags.hpp"
#include "Evolution/DgSubcell/Actions/Initialize.hpp"
#include "Evolution/DgSubcell/Actions/Labels.hpp"
#include "Evolution/DgSubcell/Actions/ReconstructionCommunication.hpp"
#include "Evolution/DgSubcell/Actions/SelectNumericalMethod.hpp"
#include "Evolution/DgSubcell/Actions/TakeTimeStep.hpp"
#include "Evolution/DgSubcell/Actions/TciAndRollback.hpp"
#include "Evolution/DgSubcell/Actions/TciAndSwitchToDg.hpp"
#include "Evolution/DgSubcell/BackgroundGrVars.hpp"
#include "Evolution/DgSubcell/CartesianFluxDivergence.hpp"
#include "Evolution/DgSubcell/CellCenteredFlux.hpp"
#include "Evolution/DgSubcell/ComputeBoundaryTerms.hpp"
#include "Evolution/DgSubcell/CorrectPackagedData.hpp"
#include "Evolution/DgSubcell/GetTciDecision.hpp"
#include "Evolution/DgSubcell/NeighborReconstructedFaceSolution.hpp"
#include "Evolution/DgSubcell/NeighborTciDecision.hpp"
#include "Evolution/DgSubcell/PerssonTci.hpp"
#include "Evolution/DgSubcell/PrepareNeighborData.hpp"
#include "Evolution/DgSubcell/SetInterpolators.hpp"
#include "Evolution/DgSubcell/Tags/MethodOrder.hpp"
#include "Evolution/DgSubcell/Tags/ObserverCoordinates.hpp"
#include "Evolution/DgSubcell/Tags/ObserverMesh.hpp"
#include "Evolution/DgSubcell/Tags/TciStatus.hpp"
#include "Evolution/DgSubcell/TwoMeshRdmpTci.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ApplyBoundaryCorrections.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivative.hpp"
#include "Evolution/DiscontinuousGalerkin/BackgroundGrVars.hpp"
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/Mortars.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Minmod.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Tags.hpp"
#include "Evolution/Imex/Actions/DoImplicitStep.hpp"
#include "Evolution/Imex/Actions/RecordTimeStepperData.hpp"
#include "Evolution/Imex/ImplicitDenseOutput.hpp"
#include "Evolution/Imex/Initialize.hpp"
#include "Evolution/Initialization/ConservativeSystem.hpp"
#include "Evolution/Initialization/DgDomain.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Initialization/Limiter.hpp"
#include "Evolution/Initialization/SetVariables.hpp"
#include "Evolution/Systems/ForceFree/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/ForceFree/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/ForceFree/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/ForceFree/BoundaryCorrections/RegisterDerived.hpp"
#include "Evolution/Systems/ForceFree/Constraints.hpp"
#include "Evolution/Systems/ForceFree/ElectricCurrentDensity.hpp"
#include "Evolution/Systems/ForceFree/ElectromagneticVariables.hpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/RegisterDerivedWithCharm.hpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/ForceFree/Fluxes.hpp"
#include "Evolution/Systems/ForceFree/ImposeMhdConditionInsideNs.hpp"
#include "Evolution/Systems/ForceFree/MaskNeutronStarInterior.hpp"
#include "Evolution/Systems/ForceFree/NsInteriorSpatialVelocity.hpp"
#include "Evolution/Systems/ForceFree/Subcell/GhostData.hpp"
#include "Evolution/Systems/ForceFree/Subcell/NeighborPackagedData.hpp"
#include "Evolution/Systems/ForceFree/Subcell/SetInitialRdmpData.hpp"
#include "Evolution/Systems/ForceFree/Subcell/SwapGrTags.hpp"
#include "Evolution/Systems/ForceFree/Subcell/SwapMask.hpp"
#include "Evolution/Systems/ForceFree/Subcell/TciOnDgGrid.hpp"
#include "Evolution/Systems/ForceFree/Subcell/TciOnFdGrid.hpp"
#include "Evolution/Systems/ForceFree/Subcell/TciOptions.hpp"
#include "Evolution/Systems/ForceFree/Subcell/TimeDerivative.hpp"
#include "Evolution/Systems/ForceFree/System.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "Evolution/Tags/Filter.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/Actions/RegisterWithObservers.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/LinearOperators/ExponentialFilter.hpp"
#include "Options/Options.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Options/String.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseControl/CheckpointAndExitAfterWallclock.hpp"
#include "Parallel/PhaseControl/ExecutePhaseChange.hpp"
#include "Parallel/PhaseControl/Factory.hpp"
#include "Parallel/PhaseControl/PhaseChange.hpp"
#include "Parallel/PhaseControl/VisitAndReturn.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Protocols/RegistrationMetavariables.hpp"
#include "ParallelAlgorithms/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Actions/AddSimpleTags.hpp"
#include "ParallelAlgorithms/Actions/FilterAction.hpp"
#include "ParallelAlgorithms/Actions/InitializeItems.hpp"
#include "ParallelAlgorithms/Actions/LimiterActions.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Events/Factory.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/DenseTriggers/Factory.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Completion.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/ElementInitInterpPoints.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InitializeInterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/ObserveSurfaceData.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/ObserveTimeSeriesOnSurface.hpp"
#include "ParallelAlgorithms/Interpolation/Events/InterpolateWithoutInterpComponent.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Interpolator.hpp"
#include "ParallelAlgorithms/Interpolation/PointInfoTag.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/Sphere.hpp"
#include "PointwiseFunctions/AnalyticData/ForceFree/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/ForceFree/Factory.hpp"
#include "PointwiseFunctions/AnalyticSolutions/ForceFree/Factory.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/DetAndInverseSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Surfaces/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Time/Actions/AdvanceTime.hpp"
#include "Time/Actions/CleanHistory.hpp"
#include "Time/Actions/RecordTimeStepperData.hpp"
#include "Time/Actions/SelfStartActions.hpp"
#include "Time/Actions/UpdateU.hpp"
#include "Time/ChangeSlabSize/Action.hpp"
#include "Time/StepChoosers/Factory.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/TimeSequence.hpp"
#include "Time/TimeSteppers/Factory.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Time/Triggers/TimeTriggers.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"
#include "Utilities/MemoryHelpers.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
namespace Parallel {
template <typename Metavariables>
class CProxy_GlobalCache;
}  // namespace Parallel
/// \endcond

struct EvolutionMetavars {
  static constexpr size_t volume_dim = 3;
  using system = ForceFree::System;
  using temporal_id = Tags::TimeStepId;
  using TimeStepperBase = ImexTimeStepper;

  static constexpr bool local_time_stepping =
      TimeStepperBase::local_time_stepping;
  static constexpr bool use_dg_element_collection = false;

  static constexpr bool imex_time_stepping = TimeStepperBase::imex;

  // The use_dg_subcell flag controls whether to use "standard" limiting (false)
  // or a DG-FD hybrid scheme (true).
  static constexpr bool use_dg_subcell = true;

  using initial_data_list = tmpl::append<ForceFree::Solutions::all_solutions,
                                         ForceFree::AnalyticData::all_data>;

  using limiter = Tags::Limiter<Limiters::Minmod<
      3, tmpl::list<ForceFree::Tags::TildeE, ForceFree::Tags::TildeB,
                    ForceFree::Tags::TildeQ>>>;

  using analytic_variables_tags = typename system::variables_tag::tags_list;

  using analytic_compute = evolution::Tags::AnalyticSolutionsCompute<
      volume_dim, analytic_variables_tags, use_dg_subcell, initial_data_list>;

  using error_compute = Tags::ErrorsCompute<analytic_variables_tags>;

  using error_tags = db::wrap_tags_in<Tags::Error, analytic_variables_tags>;

  using observe_fields = tmpl::append<
      typename system::variables_tag::tags_list, error_tags,
      tmpl::conditional_t<
          use_dg_subcell,
          tmpl::list<evolution::dg::subcell::Tags::TciStatusCompute<volume_dim>,
                     evolution::dg::subcell::Tags::ObserverCoordinatesCompute<
                         volume_dim, Frame::ElementLogical>,
                     evolution::dg::subcell::Tags::ObserverCoordinatesCompute<
                         volume_dim, Frame::Grid>,
                     evolution::dg::subcell::Tags::ObserverCoordinatesCompute<
                         volume_dim, Frame::Inertial>>,
          tmpl::list<::Events::Tags::ObserverCoordinatesCompute<
                         volume_dim, Frame::ElementLogical>,
                     ::Events::Tags::ObserverCoordinatesCompute<volume_dim,
                                                                Frame::Grid>,
                     ::Events::Tags::ObserverCoordinatesCompute<
                         volume_dim, Frame::Inertial>>>,
      tmpl::list<gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
                 ForceFree::Tags::ElectricFieldCompute,
                 ForceFree::Tags::MagneticFieldCompute,
                 ForceFree::Tags::ChargeDensityCompute,
                 ForceFree::Tags::ElectricCurrentDensityCompute,
                 ForceFree::Tags::ElectricFieldDotMagneticFieldCompute,
                 ForceFree::Tags::MagneticDominanceViolationCompute,
                 ForceFree::Tags::JouleHeatingCompute,
                 ForceFree::Tags::NsInteriorSpatialVelocity>>;

  using non_tensor_compute_tags = tmpl::append<
      tmpl::conditional_t<
          use_dg_subcell,
          tmpl::list<
              evolution::dg::subcell::Tags::ObserverMeshCompute<volume_dim>,
              evolution::dg::subcell::Tags::ObserverInverseJacobianCompute<
                  volume_dim, Frame::ElementLogical, Frame::Inertial>,
              evolution::dg::subcell::Tags::
                  ObserverJacobianAndDetInvJacobianCompute<
                      volume_dim, Frame::ElementLogical, Frame::Inertial>>,
          tmpl::list<::Events::Tags::ObserverMeshCompute<volume_dim>,
                     ::Events::Tags::ObserverInverseJacobianCompute<
                         volume_dim, Frame::ElementLogical, Frame::Inertial>,
                     ::Events::Tags::ObserverJacobianCompute<
                         volume_dim, Frame::ElementLogical, Frame::Inertial>,
                     ::Events::Tags::ObserverDetInvJacobianCompute<
                         Frame::ElementLogical, Frame::Inertial>>>,
      tmpl::list<analytic_compute, error_compute>>;

  struct TotalMagneticFluxOnUpperHemisphere
      : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    using temporal_id = ::Tags::Time;

    using vars_to_interpolate_to_target =
        tmpl::list<ForceFree::Tags::TildeB,
                   //    domain::Tags::Coordinates<3, Frame::Inertial>,
                   gr::Tags::SpatialMetric<DataVector, 3, Frame::Inertial>>;

    using compute_items_on_source = tmpl::list<>;

    using compute_items_on_target = tmpl::list<
        gr::Tags::DetAndInverseSpatialMetricCompute<DataVector, 3,
                                                    Frame::Inertial>,
        gr::Tags::SqrtDetSpatialMetricCompute<DataVector, 3, Frame::Inertial>,
        ylm::Tags::OneOverOneFormMagnitudeCompute<DataVector, 3,
                                                  Frame::Inertial>,
        ylm::Tags::UnitNormalOneFormCompute<Frame::Inertial>,
        //
        ForceFree::Tags::MagneticFluxCompute,
        //
        gr::surfaces::Tags::AreaElementCompute<Frame::Inertial>,
        gr::surfaces::Tags::SurfaceIntegralCompute<
            ForceFree::Tags::MagneticFlux, Frame::Inertial>>;

    using compute_target_points =
        intrp::TargetPoints::Sphere<TotalMagneticFluxOnUpperHemisphere,
                                    ::Frame::Inertial>;

    using post_interpolation_callbacks =
        tmpl::list<intrp::callbacks::ObserveTimeSeriesOnSurface<
            tmpl::list<gr::surfaces::Tags::SurfaceIntegral<
                ForceFree::Tags::MagneticFlux, Frame::Inertial>>,
            TotalMagneticFluxOnUpperHemisphere>>;

    template <typename Metavariables>
    using interpolating_component =
        typename Metavariables::dg_element_array_component;
  };

  // Observe Poynting flux S^in_i  on a sphere (used for Pulsar tests)
  struct PoyntingFluxOnSphere
      : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    using temporal_id = ::Tags::Time;

    using vars_to_interpolate_to_target =
        tmpl::list<ForceFree::Tags::TildeE, ForceFree::Tags::TildeB,
                   gr::Tags::SpatialMetric<DataVector, 3, Frame::Inertial>>;

    using compute_items_on_source = tmpl::list<>;

    using compute_items_on_target = tmpl::list<
        gr::Tags::DetAndInverseSpatialMetricCompute<DataVector, 3,
                                                    Frame::Inertial>,
        gr::Tags::SqrtDetSpatialMetricCompute<DataVector, 3, Frame::Inertial>,
        ylm::Tags::OneOverOneFormMagnitudeCompute<DataVector, 3,
                                                  Frame::Inertial>,
        ylm::Tags::UnitNormalOneFormCompute<Frame::Inertial>,
        ylm::Tags::UnitNormalVectorCompute<Frame::Inertial>,
        ForceFree::Tags::PoyntingCovectorCompute,
        ForceFree::Tags::PoyntingFluxCompute,
        gr::surfaces::Tags::AreaElementCompute<Frame::Inertial>,
        gr::surfaces::Tags::SurfaceIntegralCompute<
            ForceFree::Tags::PoyntingFlux, Frame::Inertial>>;

    using compute_target_points =
        intrp::TargetPoints::Sphere<PoyntingFluxOnSphere, ::Frame::Inertial>;

    using post_interpolation_callbacks =
        tmpl::list<intrp::callbacks::ObserveSurfaceData<
                       tmpl::list<ForceFree::Tags::PoyntingFlux>,
                       PoyntingFluxOnSphere, ::Frame::Inertial>,
                   intrp::callbacks::ObserveTimeSeriesOnSurface<
                       tmpl::list<gr::surfaces::Tags::SurfaceIntegral<
                           ForceFree::Tags::PoyntingFlux, Frame::Inertial>>,
                       PoyntingFluxOnSphere>>;

    template <typename Metavariables>
    using interpolating_component =
        typename Metavariables::dg_element_array_component;
  };

  using interpolation_target_tags =
      tmpl::list<TotalMagneticFluxOnUpperHemisphere, PoyntingFluxOnSphere>;

  using total_magnetic_flux_interpolator_source_vars =
      tmpl::list<ForceFree::Tags::TildeB,
                 gr::Tags::SpatialMetric<DataVector, 3, Frame::Inertial>>;

  using poynting_flux_interpolator_source_vars =
      tmpl::list<ForceFree::Tags::TildeE, ForceFree::Tags::TildeB,
                 gr::Tags::SpatialMetric<DataVector, 3, Frame::Inertial>>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<ForceFree::BoundaryConditions::BoundaryCondition,
                   ForceFree::BoundaryConditions::standard_boundary_conditions>,
        tmpl::pair<DenseTrigger, DenseTriggers::standard_dense_triggers>,
        tmpl::pair<DomainCreator<volume_dim>, domain_creators<volume_dim>>,
        tmpl::pair<Event,
                   tmpl::flatten<tmpl::list<
                       // For interpolations
                       intrp::Events::InterpolateWithoutInterpComponent<
                           3, TotalMagneticFluxOnUpperHemisphere,
                           total_magnetic_flux_interpolator_source_vars>,
                       intrp::Events::InterpolateWithoutInterpComponent<
                           3, PoyntingFluxOnSphere,
                           poynting_flux_interpolator_source_vars>,

                       Events::Completion,
                       dg::Events::field_observations<
                           volume_dim, observe_fields, non_tensor_compute_tags>,
                       Events::time_events<system>>>>,
        tmpl::pair<evolution::initial_data::InitialData, initial_data_list>,
        tmpl::pair<LtsTimeStepper, TimeSteppers::lts_time_steppers>,
        tmpl::pair<PhaseChange, PhaseControl::factory_creatable_classes>,
        tmpl::pair<StepChooser<StepChooserUse::LtsStep>,
                   StepChoosers::standard_step_choosers<system>>,
        tmpl::pair<
            StepChooser<StepChooserUse::Slab>,
            StepChoosers::standard_slab_choosers<system, local_time_stepping>>,
        tmpl::pair<TimeSequence<double>,
                   TimeSequences::all_time_sequences<double>>,
        tmpl::pair<TimeSequence<std::uint64_t>,
                   TimeSequences::all_time_sequences<std::uint64_t>>,
        tmpl::pair<TimeStepper, TimeSteppers::time_steppers>,
        tmpl::pair<ImexTimeStepper, TimeSteppers::imex_time_steppers>,
        tmpl::pair<Trigger, tmpl::append<Triggers::logical_triggers,
                                         Triggers::time_triggers>>>;
  };

  struct SubcellOptions {
    static constexpr bool subcell_enabled = use_dg_subcell;
    static constexpr bool subcell_enabled_at_external_boundary = true;

    // We send `ghost_zone_size` cell-centered grid points for variable
    // reconstruction, of which we need `ghost_zone_size-1` for reconstruction
    // to the internal side of the element face, and `ghost_zone_size` for
    // reconstruction to the external side of the element face.
    template <typename DbTagsList>
    static constexpr size_t ghost_zone_size(
        const db::DataBox<DbTagsList>& box) {
      return db::get<ForceFree::fd::Tags::Reconstructor>(box).ghost_zone_size();
    }

    using DgComputeSubcellNeighborPackagedData =
        ForceFree::subcell::NeighborPackagedData;

    using GhostVariables = ForceFree::subcell::GhostVariables;
  };

  using observed_reduction_data_tags =
      observers::collect_reduction_data_tags<tmpl::flatten<tmpl::list<
          tmpl::at<typename factory_creation::factory_classes, Event>>>>;

  using dg_step_actions = tmpl::flatten<tmpl::list<
      Actions::MutateApply<
          evolution::dg::BackgroundGrVars<system, EvolutionMetavars, true>>,
      evolution::dg::Actions::ComputeTimeDerivative<
          volume_dim, system, AllStepChoosers, local_time_stepping,
          use_dg_element_collection>,
      tmpl::conditional_t<
          local_time_stepping,
          tmpl::list<evolution::Actions::RunEventsAndDenseTriggers<tmpl::list<
                         evolution::dg::ApplyBoundaryCorrections<
                             local_time_stepping, system, volume_dim, true>,
                         tmpl::conditional_t<imex_time_stepping,
                                             imex::ImplicitDenseOutput<system>,
                                             tmpl::list<>>>>,
                     evolution::dg::Actions::ApplyLtsBoundaryCorrections<
                         system, volume_dim, false, use_dg_element_collection>>,
          tmpl::list<
              evolution::dg::Actions::ApplyBoundaryCorrectionsToTimeDerivative<
                  system, volume_dim, false, use_dg_element_collection>,
              Actions::RecordTimeStepperData<system>,
              tmpl::conditional_t<imex_time_stepping,
                                  imex::Actions::RecordTimeStepperData<system>,
                                  tmpl::list<>>,
              evolution::Actions::RunEventsAndDenseTriggers<tmpl::list<>>,
              Actions::UpdateU<system>>>,

      // Manually check the E dot B constraint.
      ForceFree::Actions::ObserveEdotB<true>,

      // implicit step
      tmpl::conditional_t<imex_time_stepping,
                          imex::Actions::DoImplicitStep<system>, tmpl::list<>>,

      ForceFree::Actions::ObserveEdotB<false>,

      // Interior BC or heck inside the horizon, depending on the ID type
      Actions::MutateApply<ForceFree::ImposeMhdConditionInsideNs>,

      Actions::CleanHistory<system, local_time_stepping>,

      Limiters::Actions::SendData<EvolutionMetavars>,
      Limiters::Actions::Limit<EvolutionMetavars>,

      dg::Actions::Filter<
          Filters::Exponential<0>,
          tmpl::list<ForceFree::Tags::TildeE, ForceFree::Tags::TildeB,
                     ForceFree::Tags::TildePsi, ForceFree::Tags::TildePhi,
                     ForceFree::Tags::TildeQ>>

      >>;

  using dg_subcell_step_actions = tmpl::flatten<tmpl::list<
      evolution::dg::subcell::Actions::SelectNumericalMethod,

      Actions::Label<evolution::dg::subcell::Actions::Labels::BeginDg>,
      Actions::MutateApply<
          evolution::dg::BackgroundGrVars<system, EvolutionMetavars, true>>,
      evolution::dg::Actions::ComputeTimeDerivative<
          volume_dim, system, AllStepChoosers, local_time_stepping>,
      evolution::dg::Actions::ApplyBoundaryCorrectionsToTimeDerivative<
          system, volume_dim, false>,

      tmpl::list<Actions::RecordTimeStepperData<system>,
                 imex::Actions::RecordTimeStepperData<system>,

                 evolution::Actions::RunEventsAndDenseTriggers<
                     tmpl::list<imex::ImplicitDenseOutput<system>>>,

                 Actions::UpdateU<system>>,

      // implicit step
      imex::Actions::DoImplicitStep<system>,

      // Interior BC
      Actions::MutateApply<ForceFree::ImposeMhdConditionInsideNs>,

      evolution::dg::subcell::Actions::TciAndRollback<
          ForceFree::subcell::TciOnDgGrid>,

      Actions::Goto<evolution::dg::subcell::Actions::Labels::EndOfSolvers>,

      Actions::Label<evolution::dg::subcell::Actions::Labels::BeginSubcell>,
      //   at the beginning : compute all
      Actions::MutateApply<evolution::dg::subcell::BackgroundGrVars<
          system, EvolutionMetavars, true, false>>,

      // cell-centered flux
      Actions::MutateApply<evolution::dg::subcell::fd::CellCenteredFlux<
          system, ForceFree::Fluxes, volume_dim, false>>,

      evolution::dg::subcell::Actions::SendDataForReconstruction<
          volume_dim, ForceFree::subcell::GhostVariables, local_time_stepping>,
      evolution::dg::subcell::Actions::ReceiveDataForReconstruction<volume_dim>,
      Actions::Label<
          evolution::dg::subcell::Actions::Labels::BeginSubcellAfterDgRollback>,
      //   PUt background gr here
      //   only update subcell GR vars when we did rollback, check DIDROLLBACK
      //   tag. compute all.
      Actions::MutateApply<evolution::dg::subcell::BackgroundGrVars<
          system, EvolutionMetavars, true, true>>,

      Actions::MutateApply<ForceFree::subcell::SwapGrTags>,
      Actions::MutateApply<ForceFree::subcell::SwapMask>,

      // cell-centered flux
      Actions::MutateApply<evolution::dg::subcell::fd::CellCenteredFlux<
          system, ForceFree::Fluxes, volume_dim, true>>,

      evolution::dg::subcell::fd::Actions::TakeTimeStep<
          ForceFree::subcell::TimeDerivative>,
      Actions::RecordTimeStepperData<system>,
      imex::Actions::RecordTimeStepperData<system>,

      evolution::Actions::RunEventsAndDenseTriggers<
          tmpl::list<imex::ImplicitDenseOutput<system>>>,

      Actions::UpdateU<system>,

      // implicit step
      imex::Actions::DoImplicitStep<system>,

      // Interior BC
      Actions::MutateApply<ForceFree::ImposeMhdConditionInsideNs>,

      evolution::dg::subcell::Actions::TciAndSwitchToDg<
          ForceFree::subcell::TciOnFdGrid>,
      Actions::MutateApply<ForceFree::subcell::SwapGrTags>,
      Actions::MutateApply<ForceFree::subcell::SwapMask>,
      Actions::Label<evolution::dg::subcell::Actions::Labels::EndOfSolvers>>>;

  using step_actions =
      tmpl::conditional_t<use_dg_subcell, dg_subcell_step_actions,
                          dg_step_actions>;

  using const_global_cache_tags = tmpl::list<
      evolution::initial_data::Tags::InitialData,
      tmpl::conditional_t<use_dg_subcell,
                          tmpl::list<ForceFree::fd::Tags::Reconstructor,
                                     ForceFree::subcell::Tags::TciOptions>,
                          tmpl::list<>>,
      ForceFree::Tags::KappaPsi, ForceFree::Tags::KappaPhi,
      ForceFree::Tags::ParallelConductivity>;

  using dg_registration_list =
      tmpl::list<observers::Actions::RegisterEventsWithObservers>;

  using initialization_actions = tmpl::list<
      Initialization::Actions::InitializeItems<
          Initialization::TimeStepping<EvolutionMetavars, TimeStepperBase>,
          evolution::dg::Initialization::Domain<volume_dim>,
          Initialization::TimeStepperHistory<EvolutionMetavars>>,
      Initialization::Actions::AddSimpleTags<
          evolution::dg::BackgroundGrVars<system, EvolutionMetavars, true>>,
      Initialization::Actions::ConservativeSystem<system>,

      tmpl::conditional_t<
          use_dg_subcell,
          tmpl::list<
              evolution::dg::subcell::Actions::SetSubcellGrid<volume_dim,
                                                              system, false>,
              Actions::MutateApply<
                  evolution::dg::subcell::SetInterpolators<volume_dim>>,
              Initialization::Actions::AddSimpleTags<
                  evolution::dg::subcell::BackgroundGrVars<
                      system, EvolutionMetavars, true, false>,
                  ForceFree::MaskNeutronStarInterior<EvolutionMetavars, true>>,

              Actions::MutateApply<ForceFree::subcell::SwapGrTags>,

              evolution::dg::subcell::Actions::SetAndCommunicateInitialRdmpData<
                  volume_dim, ForceFree::subcell::SetInitialRdmpData>,
              evolution::dg::subcell::Actions::ComputeAndSendTciOnInitialGrid<
                  volume_dim, system, ForceFree::subcell::TciOnFdGrid>,
              evolution::dg::subcell::Actions::SetInitialGridFromTciData<
                  volume_dim, system>,
              Actions::MutateApply<ForceFree::subcell::SwapGrTags>,
              Actions::MutateApply<ForceFree::subcell::SwapMask>>,
          tmpl::list<evolution::Initialization::Actions::SetVariables<
                         domain::Tags::Coordinates<3, Frame::ElementLogical>>,
                     Initialization::Actions::AddSimpleTags<
                         ForceFree::MaskNeutronStarInterior<EvolutionMetavars,
                                                            false>>>>,

      // note : imex::Initialize mutator needs to be executed after
      //        the TimeStepperHistory action AND grid initialization
      Initialization::Actions::InitializeItems<imex::Initialize<system>>,

      Initialization::Actions::AddComputeTags<tmpl::list<
          ForceFree::Tags::TildeESquaredCompute,
          ForceFree::Tags::TildeBSquaredCompute,
          ForceFree::Tags::TildeEDotTildeBCompute,
          ForceFree::Tags::ComputeTildeJ,
          ForceFree::Tags::NsInteriorSpatialVelocityCompute<use_dg_subcell>>>,

      Initialization::Actions::AddComputeTags<
          StepChoosers::step_chooser_compute_tags<EvolutionMetavars,
                                                  local_time_stepping>>,
      ::evolution::dg::Initialization::Mortars<volume_dim, system>,
      Initialization::Actions::Minmod<3>,

      intrp::Actions::ElementInitInterpPoints<
          intrp::Tags::InterpPointInfo<EvolutionMetavars>>,

      evolution::Actions::InitializeRunEventsAndDenseTriggers,
      Parallel::Actions::TerminatePhase>;

  using dg_element_array_component = DgElementArray<
      EvolutionMetavars,
      tmpl::list<
          Parallel::PhaseActions<Parallel::Phase::Initialization,
                                 initialization_actions>,

          Parallel::PhaseActions<Parallel::Phase::Register,
                                 tmpl::list<dg_registration_list,
                                            Parallel::Actions::TerminatePhase>>,

          Parallel::PhaseActions<
              Parallel::Phase::InitializeTimeStepperHistory,
              SelfStart::self_start_procedure<step_actions, system>>,

          Parallel::PhaseActions<
              Parallel::Phase::Evolve,
              tmpl::list<
                  evolution::Actions::RunEventsAndTriggers<local_time_stepping>,
                  Actions::ChangeSlabSize, dg_step_actions,
                  Actions::AdvanceTime,
                  PhaseControl::Actions::ExecutePhaseChange>>>>;

  struct registration
      : tt::ConformsTo<Parallel::protocols::RegistrationMetavariables> {
    using element_registrars =
        tmpl::map<tmpl::pair<dg_element_array_component, dg_registration_list>>;
  };

  using component_list = tmpl::flatten<tmpl::list<
      observers::Observer<EvolutionMetavars>,
      observers::ObserverWriter<EvolutionMetavars>,
      tmpl::transform<interpolation_target_tags,
                      tmpl::bind<intrp::InterpolationTarget,
                                 tmpl::pin<EvolutionMetavars>, tmpl::_1>>,
      dg_element_array_component>>;

  static constexpr Options::String help{
      "Evolve the GRFFE system with divergence cleaning.\n"};

  static constexpr std::array<Parallel::Phase, 5> default_phase_order{
      {Parallel::Phase::Initialization,
       Parallel::Phase::InitializeTimeStepperHistory, Parallel::Phase::Register,
       Parallel::Phase::Evolve, Parallel::Phase::Exit}};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};
