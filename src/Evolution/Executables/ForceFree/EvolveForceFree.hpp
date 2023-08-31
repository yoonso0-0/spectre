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
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
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
#include "Evolution/DiscontinuousGalerkin/Limiters/LimiterActions.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Minmod.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Tags.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTriggers/Factory.hpp"
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
#include "Evolution/Systems/ForceFree/HeckInsideHorizon.hpp"
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
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/Actions/RegisterWithObservers.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
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
#include "ParallelAlgorithms/Actions/InitializeItems.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Events/Factory.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Actions/RunEventsOnFailure.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Completion.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "PointwiseFunctions/AnalyticData/ForceFree/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/ForceFree/Factory.hpp"
#include "PointwiseFunctions/AnalyticSolutions/ForceFree/Factory.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Time/Actions/AdvanceTime.hpp"
#include "Time/Actions/RecordTimeStepperData.hpp"
#include "Time/Actions/SelfStartActions.hpp"
#include "Time/Actions/UpdateU.hpp"
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
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
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
  static constexpr bool local_time_stepping = false;

  // A flag that controls whether to use the Implicit-Explicit (IMEX) time
  // stepping method
  static constexpr bool imex_time_stepping = true;

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

  using observe_fields = tmpl::flatten<tmpl::push_back<
      tmpl::append<
          typename system::variables_tag::tags_list, error_tags,
          tmpl::conditional_t<use_dg_subcell,
                              tmpl::list<evolution::dg::subcell::Tags::
                                             TciStatusCompute<volume_dim>>,
                              tmpl::list<>>>,
      tmpl::conditional_t<
          use_dg_subcell,
          evolution::dg::subcell::Tags::ObserverCoordinatesCompute<volume_dim,
                                                                   Frame::Grid>,
          domain::Tags::Coordinates<volume_dim, Frame::Grid>>,
      tmpl::conditional_t<
          use_dg_subcell,
          evolution::dg::subcell::Tags::ObserverCoordinatesCompute<
              volume_dim, Frame::Inertial>,
          domain::Tags::Coordinates<volume_dim, Frame::Inertial>>,
      gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
      ForceFree::Tags::ElectricFieldCompute,
      ForceFree::Tags::MagneticFieldCompute,
      ForceFree::Tags::ChargeDensityCompute,
      ForceFree::Tags::ElectricCurrentDensityCompute,
      ForceFree::Tags::ElectricFieldDotMagneticFieldCompute,
      ForceFree::Tags::MagneticDominanceViolationCompute,
      ForceFree::Tags::JouleHeatingCompute,
      ForceFree::Tags::NsInteriorSpatialVelocity>>;

  using non_tensor_compute_tags = tmpl::list<
      tmpl::conditional_t<
          use_dg_subcell,
          evolution::dg::subcell::Tags::ObserverMeshCompute<volume_dim>,
          ::Events::Tags::ObserverMeshCompute<volume_dim>>,
      ::Events::Tags::ObserverDetInvJacobianCompute<Frame::ElementLogical,
                                                    Frame::Inertial>,
      analytic_compute, error_compute>;

  struct MagneticFluxThroughHorizon
      : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {};

  struct PoyntingFlux
      : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {};

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<ForceFree::BoundaryConditions::BoundaryCondition,
                   ForceFree::BoundaryConditions::standard_boundary_conditions>,
        tmpl::pair<DenseTrigger, DenseTriggers::standard_dense_triggers>,
        tmpl::pair<DomainCreator<volume_dim>, domain_creators<volume_dim>>,
        tmpl::pair<Event,
                   tmpl::flatten<tmpl::list<
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
        tmpl::conditional_t<
            imex_time_stepping,
            tmpl::pair<ImexTimeStepper, TimeSteppers::imex_time_steppers>,
            tmpl::pair<TimeStepper, TimeSteppers::time_steppers>>,
        tmpl::pair<Trigger, tmpl::append<Triggers::logical_triggers,
                                         Triggers::time_triggers>>>;
  };

  struct SubcellOptions {
    static constexpr bool subcell_enabled = use_dg_subcell;
    static constexpr bool subcell_enabled_at_external_boundary = false;

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
      // ForceFree::Actions::ObserveEdotB<true>,
      evolution::dg::Actions::ComputeTimeDerivative<
          volume_dim, system, AllStepChoosers, local_time_stepping>,
      tmpl::conditional_t<
          local_time_stepping,
          tmpl::list<evolution::Actions::RunEventsAndDenseTriggers<tmpl::list<
                         evolution::dg::ApplyBoundaryCorrections<
                             local_time_stepping, system, volume_dim, true>,
                         tmpl::conditional_t<imex_time_stepping,
                                             imex::ImplicitDenseOutput<system>,
                                             tmpl::list<>>>>,
                     evolution::dg::Actions::ApplyLtsBoundaryCorrections<
                         system, volume_dim, false>>,
          tmpl::list<
              evolution::dg::Actions::ApplyBoundaryCorrectionsToTimeDerivative<
                  system, volume_dim, false>,
              Actions::RecordTimeStepperData<system>,
              tmpl::conditional_t<imex_time_stepping,
                                  imex::Actions::RecordTimeStepperData,
                                  tmpl::list<>>,
              evolution::Actions::RunEventsAndDenseTriggers<tmpl::list<>>,
              Actions::UpdateU<system>>>,

      // Manually check the E dot B constraint.
      //   ForceFree::Actions::ObserveEdotB<true>,

      // implicit step
      tmpl::conditional_t<imex_time_stepping, imex::Actions::DoImplicitStep,
                          tmpl::list<>>,
      //   ForceFree::Actions::ObserveEdotB<false>,

      // After implicit solve, impose the MHD condition if applicable
      Actions::MutateApply<ForceFree::ImposeMhdConditionInsideNs>

      // Disable limiters
      //   Limiters::Actions::SendData<EvolutionMetavars>,
      //   Limiters::Actions::Limit<EvolutionMetavars>>
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
                 tmpl::conditional_t<imex_time_stepping,
                                     imex::Actions::RecordTimeStepperData,
                                     tmpl::list<>>,
                 evolution::Actions::RunEventsAndDenseTriggers<tmpl::list<>>,
                 Actions::UpdateU<system>>,

      // implicit step
      tmpl::conditional_t<imex_time_stepping, imex::Actions::DoImplicitStep,
                          tmpl::list<>>,
      // Interior BC
      Actions::MutateApply<ForceFree::ImposeMhdConditionInsideNs>,

      evolution::dg::subcell::Actions::TciAndRollback<
          ForceFree::subcell::TciOnDgGrid>,

      Actions::Goto<evolution::dg::subcell::Actions::Labels::EndOfSolvers>,

      Actions::Label<evolution::dg::subcell::Actions::Labels::BeginSubcell>,
      //   at the beginning : compute all
      Actions::MutateApply<evolution::dg::subcell::BackgroundGrVars<
          system, EvolutionMetavars, true, false>>,
      //
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
      //
      Actions::MutateApply<ForceFree::subcell::SwapGrTags>,
      Actions::MutateApply<ForceFree::subcell::SwapMask>,
      evolution::dg::subcell::fd::Actions::TakeTimeStep<
          ForceFree::subcell::TimeDerivative>,
      Actions::RecordTimeStepperData<system>,
      tmpl::conditional_t<imex_time_stepping,
                          imex::Actions::RecordTimeStepperData, tmpl::list<>>,
      evolution::Actions::RunEventsAndDenseTriggers<tmpl::list<>>,
      Actions::UpdateU<system>,

      // implicit step
      tmpl::conditional_t<imex_time_stepping, imex::Actions::DoImplicitStep,
                          tmpl::list<>>,
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
          Initialization::TimeStepping<EvolutionMetavars, local_time_stepping>,
          evolution::dg::Initialization::Domain<volume_dim>,
          Initialization::TimeStepperHistory<EvolutionMetavars>>,
      Initialization::Actions::AddSimpleTags<
          evolution::dg::BackgroundGrVars<system, EvolutionMetavars, true>>,
      Initialization::Actions::ConservativeSystem<system>,
      //   evolution::Initialization::Actions::SetVariables<
      //   domain::Tags::Coordinates<3, Frame::ElementLogical>>,

      Initialization::Actions::AddSimpleTags<
          ForceFree::MaskNeutronStarInterior<EvolutionMetavars, false>>,

      tmpl::conditional_t<
          use_dg_subcell,
          tmpl::list<
              evolution::dg::subcell::Actions::SetSubcellGrid<volume_dim,
                                                              system, false>,
              Initialization::Actions::AddSimpleTags<
                  evolution::dg::subcell::BackgroundGrVars<
                      system, EvolutionMetavars, true, false>,
                  ForceFree::MaskNeutronStarInterior<EvolutionMetavars, true>>,

              Actions::MutateApply<ForceFree::subcell::SwapGrTags>,
              Actions::MutateApply<ForceFree::subcell::SwapMask>,

              evolution::dg::subcell::Actions::SetAndCommunicateInitialRdmpData<
                  volume_dim, ForceFree::subcell::SetInitialRdmpData>,
              evolution::dg::subcell::Actions::ComputeAndSendTciOnInitialGrid<
                  volume_dim, system, ForceFree::subcell::TciOnFdGrid>,
              evolution::dg::subcell::Actions::SetInitialGridFromTciData<
                  volume_dim, system>,
              Actions::MutateApply<ForceFree::subcell::SwapGrTags>>,
          tmpl::list<evolution::Initialization::Actions::SetVariables<
              domain::Tags::Coordinates<3, Frame::ElementLogical>>>>,

      // note : imex::Initialize mutator needs to be executed after
      //        the TimeStepperHistory action
      tmpl::conditional_t<
          imex_time_stepping,
          Initialization::Actions::InitializeItems<imex::Initialize<system>>,
          tmpl::list<>>,

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
              tmpl::list<evolution::Actions::RunEventsAndTriggers,
                         Actions::ChangeSlabSize, step_actions,
                         Actions::AdvanceTime,
                         PhaseControl::Actions::ExecutePhaseChange>>>>;

  struct registration
      : tt::ConformsTo<Parallel::protocols::RegistrationMetavariables> {
    using element_registrars =
        tmpl::map<tmpl::pair<dg_element_array_component, dg_registration_list>>;
  };

  using component_list =
      tmpl::list<observers::Observer<EvolutionMetavars>,
                 observers::ObserverWriter<EvolutionMetavars>,
                 dg_element_array_component>;

  static constexpr Options::String help{
      "Evolve the GRFFE system with divergence cleaning.\n"};

  static constexpr std::array<Parallel::Phase, 5> default_phase_order{
      {Parallel::Phase::Initialization,
       Parallel::Phase::InitializeTimeStepperHistory, Parallel::Phase::Register,
       Parallel::Phase::Evolve, Parallel::Phase::Exit}};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling,
    &setup_memory_allocation_failure_reporting,
    &disable_openblas_multithreading,
    &domain::creators::register_derived_with_charm,
    &domain::creators::time_dependence::register_derived_with_charm,
    &domain::FunctionsOfTime::register_derived_with_charm,
    &ForceFree::BoundaryCorrections::register_derived_with_charm,
    &ForceFree::fd::register_derived_with_charm,
    &register_factory_classes_with_charm<metavariables>};

static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions, &enable_segfault_handler};
