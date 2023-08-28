# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ForceFree)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Characteristics.cpp
  Constraints.cpp
  ElectricCurrentDensity.cpp
  ElectromagneticVariables.cpp
  Fluxes.cpp
  ImposeMhdConditionInsideNs.cpp
  NsInteriorSpatialVelocity.cpp
  Sources.cpp
  TimeDerivativeTerms.cpp
  VolumeTermsInstantiation.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Characteristics.hpp
  Constraints.hpp
  ElectricCurrentDensity.hpp
  ElectromagneticVariables.hpp
  Fluxes.hpp
  InitialGuess.hpp
  ImposeMhdConditionInsideNs.hpp
  MaskNeutronStarInterior.hpp
  NsInteriorSpatialVelocity.hpp
  PoyntingVector.hpp
  Sources.hpp
  System.hpp
  Tags.hpp
  TimeDerivativeTerms.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Boost::boost
  CoordinateMaps
  DataStructures
  DiscontinuousGalerkin
  Domain
  DomainBoundaryConditions
  DomainStructure
  DgSubcell
  ErrorHandling
  Evolution
  FiniteDifference
  ForceFreeAnalyticData
  ForceFreeSolutions
  GeneralRelativity
  Hydro
  Options
  Parallel
  Spectral
  Utilities
  )

add_subdirectory(BoundaryConditions)
add_subdirectory(BoundaryCorrections)
add_subdirectory(FiniteDifference)
add_subdirectory(Imex)
add_subdirectory(Subcell)
