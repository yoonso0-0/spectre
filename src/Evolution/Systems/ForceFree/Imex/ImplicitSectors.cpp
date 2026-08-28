// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/Imex/ImplicitSectors.hpp"

#include "Evolution/Imex/CleanHistory.hpp"
#include "Evolution/Imex/CleanHistory.tpp"
#include "Evolution/Imex/SolveImplicitSector.hpp"
#include "Evolution/Imex/SolveImplicitSector.tpp"
#include "Evolution/Systems/ForceFree/System.hpp"

template struct imex::SolveImplicitSector<ForceFree::System::variables_tag,
                                          ForceFree::Imex::ParallelCurrent>;
template struct imex::CleanHistory<ForceFree::System>;
// template struct imex::Initialize<
// ForceFree::System, tmpl::list<ForceFree::System::implicit_sectors>>;
