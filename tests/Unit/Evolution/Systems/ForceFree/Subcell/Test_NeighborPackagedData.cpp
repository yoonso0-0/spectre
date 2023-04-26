// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/InterfaceLogicalCoordinates.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DgSubcell/Actions/BackgroundGrVars.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/Systems/ForceFree/BoundaryCorrections/Rusanov.hpp"
#include "Evolution/Systems/ForceFree/ElectricCurrentDensity.hpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/MonotonisedCentral.hpp"
#include "Evolution/Systems/ForceFree/Fluxes.hpp"
#include "Evolution/Systems/ForceFree/Subcell/NeighborPackagedData.hpp"
#include "Evolution/Systems/ForceFree/System.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/Systems/ForceFree/FiniteDifference/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/ForceFree/FastWave.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/Gsl.hpp"

#include <iostream>

namespace ForceFree {
namespace {

struct MetavariablesForTest {
  using component_list = tmpl::list<>;
  //   using initial_data_list =
  //   tmpl::list<RelativisticEuler::Solutions::TovStar>;

  //   struct factory_creation
  //       : tt::ConformsTo<Options::protocols::FactoryCreation> {
  //     using factory_classes = tmpl::map<
  //         tmpl::pair<evolution::initial_data::InitialData,
  //         initial_data_list>>;
  //   };
};

void test_neighbor_packaged_data(const size_t num_dg_pts_per_dimension,
                                 const gsl::not_null<std::mt19937*> gen) {
  // 1. create random U vector on an element and its neighbor elements
  // 2. send through reconstruction and compute FD fluxes on mortars
  // 3. feed argument variables of dg_package_data() function to
  //    the NeighborPackagedData struct and retrieve the packaged data
  // 4. check if it agrees with the expected value

  using evolved_vars_tags = typename System::variables_tag::tags_list;
  using fluxes_tags = typename Fluxes::return_tags;

  // Perform test with MC reconstruction & Rusanov riemann solver
  using ReconstructionForTest = typename fd::MonotonisedCentral;
  using BoundaryCorrectionForTest = typename BoundaryCorrections::Rusanov;

  //
  using SolutionForTest = Solutions::FastWave;
  const SolutionForTest solution{};

  // create an element and its neighbor elements
  DirectionMap<3, Neighbors<3>> element_neighbors{};
  for (size_t i = 0; i < 2 * 3; ++i) {
    element_neighbors[gsl::at(Direction<3>::all_directions(), i)] =
        Neighbors<3>{{ElementId<3>{i + 1, {}}}, {}};
  }
  const Element<3> element{ElementId<3>{0, {}}, element_neighbors};

  const auto logical_to_grid_map = ElementMap<3, Frame::Grid>{
      ElementId<3>{0},
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
          domain::CoordinateMaps::Identity<3>{})};
  const auto grid_to_inertial_map =
      domain::make_coordinate_map<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<3>{});

  // below are required for calling GrTagsForHydro::apply() to compute metric at
  // FD cell interfaces
  const double time{0.0};
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};

  // generate random U on the dg mesh and project it to subcell mesh
  const Mesh<3> dg_mesh{num_dg_pts_per_dimension, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);

  // volume vars on DG and subcell mesh
  Variables<evolved_vars_tags> evolved_vars_dg{dg_mesh.number_of_grid_points()};
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  fill_with_random_values(make_not_null(&evolved_vars_dg), gen,
                          make_not_null(&dist));
  Variables<evolved_vars_tags> volume_vars_subcell =
      evolution::dg::subcell::fd::project(evolved_vars_dg, dg_mesh,
                                          subcell_mesh.extents());

  // Create metric variables on DG mesh
  const auto dg_inertial_coords =
      grid_to_inertial_map(logical_to_grid_map(logical_coordinates(dg_mesh)),
                           time, functions_of_time);
  Variables<typename System::spacetime_variables_tag::tags_list> metric_vars_dg{
      dg_mesh.number_of_grid_points()};
  metric_vars_dg.assign_subset(solution.variables(
      dg_inertial_coords, time,
      typename System::spacetime_variables_tag::tags_list{}));

  using subcell_metric_field =
      evolution::dg::subcell::Tags::Inactive<System::spacetime_variables_tag>;

  // metric variables on the FD interfaces
  std::array<typename System::flux_spacetime_variables_tag::type, 3>
      face_centered_gr_vars{};

  // TildeJ
  const double parallel_conductivity = 1e5;
  Tags::TildeJ::type tilde_j_dg{dg_mesh.number_of_grid_points()};
  //   Tags::ComputeTildeJ::function(
  //       make_not_null(&tilde_j_dg), get<Tags::TildeQ>(evolved_vars_dg),
  //       get<Tags::TildeE>(evolved_vars_dg),
  //       get<Tags::TildeB>(evolved_vars_dg), parallel_conductivity, )

  // generate random ghost data from neighbor
  auto logical_coords_subcell = logical_coordinates(subcell_mesh);
  const ReconstructionForTest reconstructor{};
  const auto compute_random_variable = [&gen, &dist](const auto& coords) {
    Variables<evolved_vars_tags> vars{get<0>(coords).size(), 0.0};
    fill_with_random_values(make_not_null(&vars), gen, make_not_null(&dist));
    return vars;
  };
  typename evolution::dg::subcell::Tags::GhostDataForReconstruction<3>::type
      ghost_data = TestHelpers::ForceFree::fd::compute_ghost_data(
          subcell_mesh, logical_coords_subcell, element.neighbors(),
          reconstructor.ghost_zone_size(), compute_random_variable);

  // =============== Normal vector
  DirectionMap<3, std::optional<Variables<
                      tmpl::list<evolution::dg::Tags::MagnitudeOfNormal,
                                 evolution::dg::Tags::NormalCovector<3>>>>>
      normal_vectors{};
  for (const auto& direction : Direction<3>::all_directions()) {
    using inverse_spatial_metric_tag =
        typename System::inverse_spatial_metric_tag;
    const Mesh<2> face_mesh = dg_mesh.slice_away(direction.dimension());
    const auto face_logical_coords =
        interface_logical_coordinates(face_mesh, direction);
    std::unordered_map<Direction<3>, tnsr::i<DataVector, 3, Frame::Inertial>>
        unnormalized_normal_covectors{};
    tnsr::i<DataVector, 3, Frame::Inertial> unnormalized_covector{};
    const auto element_logical_to_grid_inv_jac =
        logical_to_grid_map.inv_jacobian(face_logical_coords);
    const auto grid_to_inertial_inv_jac = grid_to_inertial_map.inv_jacobian(
        logical_to_grid_map(face_logical_coords), time, functions_of_time);
    InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>
        element_logical_to_inertial_inv_jac{};
    for (size_t logical_i = 0; logical_i < 3; ++logical_i) {
      for (size_t inertial_i = 0; inertial_i < 3; ++inertial_i) {
        element_logical_to_inertial_inv_jac.get(logical_i, inertial_i) =
            element_logical_to_grid_inv_jac.get(logical_i, 0) *
            grid_to_inertial_inv_jac.get(0, inertial_i);
        for (size_t grid_i = 1; grid_i < 3; ++grid_i) {
          element_logical_to_inertial_inv_jac.get(logical_i, inertial_i) +=
              element_logical_to_grid_inv_jac.get(logical_i, grid_i) *
              grid_to_inertial_inv_jac.get(grid_i, inertial_i);
        }
      }
    }
    for (size_t i = 0; i < 3; ++i) {
      unnormalized_covector.get(i) =
          element_logical_to_inertial_inv_jac.get(direction.dimension(), i);
    }
    unnormalized_normal_covectors[direction] = unnormalized_covector;
    Variables<tmpl::list<
        inverse_spatial_metric_tag,
        evolution::dg::Actions::detail::NormalVector<3>,
        evolution::dg::Actions::detail::OneOverNormalVectorMagnitude>>
        fields_on_face{face_mesh.number_of_grid_points()};
    fields_on_face.assign_subset(solution.variables(
        grid_to_inertial_map(logical_to_grid_map(face_logical_coords), time,
                             functions_of_time),
        time, tmpl::list<inverse_spatial_metric_tag>{}));
    normal_vectors[direction] = std::nullopt;
    evolution::dg::Actions::detail::
        unit_normal_vector_and_covector_and_magnitude<System>(
            make_not_null(&normal_vectors), make_not_null(&fields_on_face),
            direction, unnormalized_normal_covectors, grid_to_inertial_map);
  }
  // =============== Normal vector

  //   tnsr::I<DataVector, 3, Frame::Inertial> inertial_coords_subcell;
  //   inertial_coords_subcell.get(0) = logical_coords_subcell.get(0);

  auto box = db::create<db::AddSimpleTags<
      domain::Tags::Element<3>, domain::Tags::Mesh<3>,
      evolution::dg::subcell::Tags::Mesh<3>, typename System::variables_tag,
      Tags::ParallelConductivity, typename System::spacetime_variables_tag,
      subcell_metric_field,
      evolution::dg::subcell::Tags::OnSubcellFaces<
          typename System::flux_spacetime_variables_tag, 3>,
      evolution::dg::subcell::Tags::GhostDataForReconstruction<3>,
      fd::Tags::Reconstructor, evolution::Tags::BoundaryCorrection<System>,
      ::Tags::Time, domain::Tags::FunctionsOfTimeInitialize,
      domain::Tags::ElementMap<3, Frame::Grid>,
      domain::CoordinateMaps::Tags::CoordinateMap<3, Frame::Grid,
                                                  Frame::Inertial>,
      evolution::dg::subcell::Tags::Coordinates<3, Frame::ElementLogical>,
      //   evolution::dg::subcell::Tags::Coordinates<3, Frame::Inertial>,
      domain::Tags::MeshVelocity<3>,
      evolution::dg::Tags::NormalCovectorAndMagnitude<3>,
      ::Tags::AnalyticSolution<SolutionForTest>,
      evolution::dg::subcell::Tags::SubcellOptions<3>>>(
      element, dg_mesh, subcell_mesh, evolved_vars_dg, parallel_conductivity,
      metric_vars_dg, typename subcell_metric_field::type{},
      face_centered_gr_vars, ghost_data,
      std::unique_ptr<fd::Reconstructor>{
          std::make_unique<ReconstructionForTest>()},
      std::unique_ptr<BoundaryCorrections::BoundaryCorrection>{
          std::make_unique<BoundaryCorrectionForTest>()},
      time, clone_unique_ptrs(functions_of_time),
      ElementMap<3, Frame::Grid>{
          ElementId<3>{0},
          domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
              domain::CoordinateMaps::Identity<3>{})},
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<3>{}),
      logical_coords_subcell,
      std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>{}, normal_vectors,
      solution,
      evolution::dg::subcell::SubcellOptions{
          1.0e-3, 1.0e-4, 1.0e-3, 1.0e-4, 4.0, 4.0, false,
          evolution::dg::subcell::fd::ReconstructionMethod::DimByDim, false,
          std::nullopt, ::fd::DerivativeOrder::Two});

  // This action needs to be called in prior since NeighborPackagedData::apply()
  // internally retrieves face-centered tensors when computing fluxes.
//   evolution::dg::subcell::Actions::BackgroundGrVars<System, false>::apply(
//       box, tuples::TaggedTuple<>{},
//       Parallel::GlobalCache<MetavariablesForTest>{}, 0, tmpl::list<>{},
//       std::unique_ptr<size_t>{}.get());

  // Compute the packaged data
  std::vector<std::pair<Direction<3>, ElementId<3>>>
      mortars_to_reconstruct_to{};
  for (const auto& [direction, neighbors] : element.neighbors()) {
    mortars_to_reconstruct_to.emplace_back(direction, *neighbors.begin());
  }
  const auto packaged_data =
      subcell::NeighborPackagedData::apply(box, mortars_to_reconstruct_to);

  for (const auto& mortar_id : mortars_to_reconstruct_to) {
    std::cout << packaged_data.at(mortar_id) << std::endl;
  }

  // Now for each directions, check that the packaged_data agrees with expected
  // values

  //
  // ...
  //
}

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.ForceFree.Subcell.NeighborPackagedData",
    "[Unit][Evolution]") {
  const size_t num_dg_pts_per_dimension = 2;
  MAKE_GENERATOR(gen);

  test_neighbor_packaged_data(num_dg_pts_per_dimension, make_not_null(&gen));
}

}  // namespace
}  // namespace ForceFree
