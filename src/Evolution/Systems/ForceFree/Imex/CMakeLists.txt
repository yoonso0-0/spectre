# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  DivergenceCleaning.cpp
  ImplicitSectors.cpp
  InitialGuess.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  DivergenceCleaning.hpp
  Imex.hpp
  ImplicitSector.hpp
  InitialGuess.hpp
  )
