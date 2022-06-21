// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/FiniteDifference/MonotonicityPreserving5.hpp"

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Evolution/Systems/NewtonianEuler/FiniteDifference/ReconstructWork.tpp"
#include "NumericalAlgorithms/FiniteDifference/MonotonicityPreserving5.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianEuler::fd {

template <size_t Dim>
Mp5Prim<Dim>::Mp5Prim(const double alpha, const double epsilon)
    : alpha_(alpha), epsilon_(epsilon) {}

template <size_t Dim>
Mp5Prim<Dim>::Mp5Prim(CkMigrateMessage* const msg) : Reconstructor<Dim>(msg) {}

template <size_t Dim>
std::unique_ptr<Reconstructor<Dim>> Mp5Prim<Dim>::get_clone() const {
  return std::make_unique<Mp5Prim>(*this);
}

template <size_t Dim>
void Mp5Prim<Dim>::pup(PUP::er& p) {
  Reconstructor<Dim>::pup(p);
  p | alpha_;
  p | epsilon_;
}

template <size_t Dim>
// NOLINTNEXTLINE
PUP::able::PUP_ID Mp5Prim<Dim>::my_PUP_ID = 0;

template <size_t Dim>
template <size_t ThermodynamicDim, typename TagsList>
void Mp5Prim<Dim>::reconstruct(
    const gsl::not_null<std::array<Variables<TagsList>, Dim>*>
        vars_on_lower_face,
    const gsl::not_null<std::array<Variables<TagsList>, Dim>*>
        vars_on_upper_face,
    const Variables<prims_tags>& volume_prims,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>& eos,
    const Element<Dim>& element,
    const FixedHashMap<
        maximum_number_of_neighbors(Dim),
        std::pair<Direction<Dim>, ElementId<Dim>>, std::vector<double>,
        boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>& neighbor_data,
    const Mesh<Dim>& subcell_mesh) const {
  reconstruct_prims_work(
      vars_on_lower_face, vars_on_upper_face,
      [this](auto upper_face_vars_ptr, auto lower_face_vars_ptr,
             const auto& volume_vars, const auto& ghost_cell_vars,
             const auto& subcell_extents, const size_t number_of_variables) {
        ::fd::reconstruction::monotonicity_preserving_5(
            upper_face_vars_ptr, lower_face_vars_ptr, volume_vars,
            ghost_cell_vars, subcell_extents, number_of_variables, alpha_,
            epsilon_);
      },
      volume_prims, eos, element, neighbor_data, subcell_mesh,
      ghost_zone_size());
}

template <size_t Dim>
template <size_t ThermodynamicDim, typename TagsList>
void Mp5Prim<Dim>::reconstruct_fd_neighbor(
    const gsl::not_null<Variables<TagsList>*> vars_on_face,
    const Variables<prims_tags>& subcell_volume_prims,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>& eos,
    const Element<Dim>& element,
    const FixedHashMap<
        maximum_number_of_neighbors(Dim),
        std::pair<Direction<Dim>, ElementId<Dim>>, std::vector<double>,
        boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>& neighbor_data,
    const Mesh<Dim>& subcell_mesh,
    const Direction<Dim> direction_to_reconstruct) const {
  reconstruct_fd_neighbor_work(
      vars_on_face,
      [this](const auto tensor_component_on_face_ptr,
             const auto& tensor_component_volume,
             const auto& tensor_component_neighbor,
             const Index<Dim>& subcell_extents,
             const Index<Dim>& ghost_data_extents,
             const Direction<Dim>& local_direction_to_reconstruct) {
        ::fd::reconstruction::reconstruct_neighbor<
            Side::Lower,
            ::fd::reconstruction::detail::MonotonicityPreserving5Reconstructor>(
            tensor_component_on_face_ptr, tensor_component_volume,
            tensor_component_neighbor, subcell_extents, ghost_data_extents,
            local_direction_to_reconstruct, alpha_, epsilon_);
      },
      [this](const auto tensor_component_on_face_ptr,
             const auto& tensor_component_volume,
             const auto& tensor_component_neighbor,
             const Index<Dim>& subcell_extents,
             const Index<Dim>& ghost_data_extents,
             const Direction<Dim>& local_direction_to_reconstruct) {
        ::fd::reconstruction::reconstruct_neighbor<
            Side::Upper,
            ::fd::reconstruction::detail::MonotonicityPreserving5Reconstructor>(
            tensor_component_on_face_ptr, tensor_component_volume,
            tensor_component_neighbor, subcell_extents, ghost_data_extents,
            local_direction_to_reconstruct, alpha_, epsilon_);
      },
      subcell_volume_prims, eos, element, neighbor_data, subcell_mesh,
      direction_to_reconstruct, ghost_zone_size());
}

template <size_t Dim>
bool operator==(const Wcns5zPrim<Dim>& lhs, const Wcns5zPrim<Dim>& rhs) {
  return lhs.alpha_ == rhs.alpha_ and lhs.epsilon_ == rhs.epsilon_;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define THERMO_DIM(data) BOOST_PP_TUPLE_ELEM(1, data)
#define TAGS_LIST(data)                                                   \
  tmpl::list<Tags::MassDensityCons, Tags::MomentumDensity<DIM(data)>,     \
             Tags::EnergyDensity, Tags::MassDensity<DataVector>,          \
             Tags::Velocity<DataVector, DIM(data)>,                       \
             Tags::SpecificInternalEnergy<DataVector>,                    \
             Tags::Pressure<DataVector>,                                  \
             ::Tags::Flux<Tags::MassDensityCons, tmpl::size_t<DIM(data)>, \
                          Frame::Inertial>,                               \
             ::Tags::Flux<Tags::MomentumDensity<DIM(data)>,               \
                          tmpl::size_t<DIM(data)>, Frame::Inertial>,      \
             ::Tags::Flux<Tags::EnergyDensity, tmpl::size_t<DIM(data)>,   \
                          Frame::Inertial>>

#define INSTANTIATION(r, data) template class Mp5Prim<DIM(data)>;
GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))
#undef INSTANTIATION

#define INSTANTIATION(r, data)                                                 \
  template void Mp5Prim<DIM(data)>::reconstruct(                               \
      gsl::not_null<std::array<Variables<TAGS_LIST(data)>, DIM(data)>*>        \
          vars_on_lower_face,                                                  \
      gsl::not_null<std::array<Variables<TAGS_LIST(data)>, DIM(data)>*>        \
          vars_on_upper_face,                                                  \
      const Variables<prims_tags>& volume_prims,                               \
      const EquationsOfState::EquationOfState<false, THERMO_DIM(data)>& eos,   \
      const Element<DIM(data)>& element,                                       \
      const FixedHashMap<                                                      \
          maximum_number_of_neighbors(DIM(data)),                              \
          std::pair<Direction<DIM(data)>, ElementId<DIM(data)>>,               \
          std::vector<double>,                                                 \
          boost::hash<std::pair<Direction<DIM(data)>, ElementId<DIM(data)>>>>& \
          neighbor_data,                                                       \
      const Mesh<DIM(data)>& subcell_mesh) const;                              \
  template void Mp5Prim<DIM(data)>::reconstruct_fd_neighbor(                   \
      gsl::not_null<Variables<TAGS_LIST(data)>*> vars_on_face,                 \
      const Variables<prims_tags>& subcell_volume_prims,                       \
      const EquationsOfState::EquationOfState<false, THERMO_DIM(data)>& eos,   \
      const Element<DIM(data)>& element,                                       \
      const FixedHashMap<                                                      \
          maximum_number_of_neighbors(DIM(data)),                              \
          std::pair<Direction<DIM(data)>, ElementId<DIM(data)>>,               \
          std::vector<double>,                                                 \
          boost::hash<std::pair<Direction<DIM(data)>, ElementId<DIM(data)>>>>& \
          neighbor_data,                                                       \
      const Mesh<DIM(data)>& subcell_mesh,                                     \
      const Direction<DIM(data)> direction_to_reconstruct) const;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3), (1, 2))

#undef INSTANTIATION
#undef TAGS_LIST
#undef THERMO_DIM
#undef DIM
}  // namespace NewtonianEuler::fd
