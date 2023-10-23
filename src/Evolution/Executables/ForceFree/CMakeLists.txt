# Distributed under the MIT License.
# See LICENSE.txt for details.

set(EXECUTABLE "EvolveForceFree")

add_spectre_executable(
  ${EXECUTABLE}
  EXCLUDE_FROM_ALL
  EvolveForceFree.cpp
  )

target_link_libraries(
  ${EXECUTABLE}
  PRIVATE
  Actions
  Charmxx::main
  DiscontinuousGalerkin
  DomainCreators
  EventsAndDenseTriggers
  EventsAndTriggers
  Evolution
  ForceFree
  ForceFreeAnalyticData
  ForceFreeSolutions
  GeneralRelativitySolutions
  GrSurfaces
  IO
  Imex
  Informer
  Interpolation
  LinearOperators
  Options
  Parallel
  ParallelInterpolation
  PhaseControl
  Time
  Utilities
  )
