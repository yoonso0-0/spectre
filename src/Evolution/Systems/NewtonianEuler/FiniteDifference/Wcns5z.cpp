// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/FiniteDifference/Wcns5z.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <tuple>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Domain/Structure/Side.hpp"
#include "Evolution/Systems/NewtonianEuler/FiniteDifference/ReconstructWork.tpp"
#include "Evolution/Systems/NewtonianEuler/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "NumericalAlgorithms/FiniteDifference/FallbackReconstructorType.hpp"
#include "NumericalAlgorithms/FiniteDifference/Wcns5z.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianEuler::fd {

template <size_t Dim>
Wcns5zPrim<Dim>::Wcns5zPrim(
    const size_t nonlinear_weight_exponent, const double epsilon,
    const ::fd::reconstruction::FallbackReconstructorType
        low_order_recons_to_fallback,
    const size_t max_number_of_extrema)
    : nonlinear_weight_exponent_(nonlinear_weight_exponent),
      epsilon_(epsilon),
      low_order_recons_to_fallback_(low_order_recons_to_fallback),
      max_number_of_extrema_(max_number_of_extrema) {
  std::tie(reconstruct_, reconstruct_lower_neighbor_,
           reconstruct_upper_neighbor_) =
      ::fd::reconstruction::wcns5z_function_pointers<Dim>(
          nonlinear_weight_exponent_, low_order_recons_to_fallback_);
}

template <size_t Dim>
Wcns5zPrim<Dim>::Wcns5zPrim(CkMigrateMessage* const msg)
    : Reconstructor<Dim>(msg) {}

template <size_t Dim>
std::unique_ptr<Reconstructor<Dim>> Wcns5zPrim<Dim>::get_clone() const {
  return std::make_unique<Wcns5zPrim>(*this);
}

template <size_t Dim>
void Wcns5zPrim<Dim>::pup(PUP::er& p) {
  Reconstructor<Dim>::pup(p);
  p | nonlinear_weight_exponent_;
  p | epsilon_;
  p | low_order_recons_to_fallback_;
  p | max_number_of_extrema_;
  if (p.isUnpacking()) {
    std::tie(reconstruct_, reconstruct_lower_neighbor_,
             reconstruct_upper_neighbor_) =
        ::fd::reconstruction::wcns5z_function_pointers<Dim>(
            nonlinear_weight_exponent_, low_order_recons_to_fallback_);
  }
}
template <size_t Dim>
// NOLINTNEXTLINE
PUP::able::PUP_ID Wcns5zPrim<Dim>::my_PUP_ID = 0;

template <size_t Dim>
template <size_t ThermodynamicDim, typename TagsList>
void Wcns5zPrim<Dim>::reconstruct(
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
        reconstruct_(upper_face_vars_ptr, lower_face_vars_ptr, volume_vars,
                     ghost_cell_vars, subcell_extents, number_of_variables,
                     epsilon_, max_number_of_extrema_);
      },
      volume_prims, eos, element, neighbor_data, subcell_mesh,
      ghost_zone_size());
}

template <size_t Dim>
template <size_t ThermodynamicDim, typename TagsList>
void Wcns5zPrim<Dim>::reconstruct_fd_neighbor(
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
        reconstruct_lower_neighbor_(
            tensor_component_on_face_ptr, tensor_component_volume,
            tensor_component_neighbor, subcell_extents, ghost_data_extents,
            local_direction_to_reconstruct, epsilon_, max_number_of_extrema_);
      },
      [this](const auto tensor_component_on_face_ptr,
             const auto& tensor_component_volume,
             const auto& tensor_component_neighbor,
             const Index<Dim>& subcell_extents,
             const Index<Dim>& ghost_data_extents,
             const Direction<Dim>& local_direction_to_reconstruct) {
        reconstruct_upper_neighbor_(
            tensor_component_on_face_ptr, tensor_component_volume,
            tensor_component_neighbor, subcell_extents, ghost_data_extents,
            local_direction_to_reconstruct, epsilon_, max_number_of_extrema_);
      },
      subcell_volume_prims, eos, element, neighbor_data, subcell_mesh,
      direction_to_reconstruct, ghost_zone_size());
}

template <size_t Dim>
bool operator==(const Wcns5zPrim<Dim>& lhs, const Wcns5zPrim<Dim>& rhs) {
  // Don't check function pointers since they are set from
  // nonlinear_weight_exponent_
  return lhs.nonlinear_weight_exponent_ == rhs.nonlinear_weight_exponent_ and
         lhs.epsilon_ == rhs.epsilon_;
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

#define INSTANTIATION(r, data)                               \
  template class Wcns5zPrim<DIM(data)>;                      \
  template bool operator==(const Wcns5zPrim<DIM(data)>& lhs, \
                           const Wcns5zPrim<DIM(data)>& rhs);
GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))
#undef INSTANTIATION

#define INSTANTIATION(r, data)                                                 \
  template void Wcns5zPrim<DIM(data)>::reconstruct(                            \
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
  template void Wcns5zPrim<DIM(data)>::reconstruct_fd_neighbor(                \
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
