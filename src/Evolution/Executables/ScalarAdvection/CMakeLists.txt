# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  ScalarAdvection
  ScalarAdvectionSolutions
  DiscontinuousGalerkin
  DomainCreators
  EventsAndTriggers
  Evolution
  IO
  Informer
  Limiters
  LinearOperators
  Options
  Parallel
  PhaseControl
  Time
  Utilities
  )

function(add_scalar_advection_executable INITIAL_DATA_NAME DIM INITIAL_DATA)
  add_spectre_parallel_executable(
    "EvolveScalarAdvection${INITIAL_DATA_NAME}${DIM}D"
    EvolveScalarAdvection
    Evolution/Executables/ScalarAdvection
    "EvolutionMetavars<${DIM},${INITIAL_DATA}>"
    "${LIBS_TO_LINK}"
    )
endfunction(add_scalar_advection_executable)

function(add_1D_executables)
  add_scalar_advection_executable(
    Sinusoid
    1
    ScalarAdvection::Solutions::Sinusoid
    )
  add_scalar_advection_executable(
    Krivodonova
    1
    ScalarAdvection::Solutions::Krivodonova
    )
endfunction(add_1D_executables)

add_1D_executables()
