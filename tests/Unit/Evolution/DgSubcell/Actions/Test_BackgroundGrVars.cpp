// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <random>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Domain/Creators/TimeDependence/UniformTranslation.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Actions/BackgroundGrVars.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/DgSubcell/Tags/OnSubcellFaces.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/GlobalCache.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/TovStar.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Time/Tags.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

#include <iostream>

namespace {

enum InitialDataType { Runtime, CompileTime };

struct MetavariablesForTest {
  using component_list = tmpl::list<>;
  using initial_data_list = tmpl::list<RelativisticEuler::Solutions::TovStar>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<evolution::initial_data::InitialData, initial_data_list>>;
  };
};

struct SystemForTest {
  static constexpr size_t volume_dim = 3;

  // A disparate set of GR variables were chosen here to make sure that the
  // action allocates and assigns metric variables without missing any tags
  using spacetime_variables_tag =
      ::Tags::Variables<tmpl::list<gr::Tags::Lapse<>, gr::Tags::Shift<3>>>;
  using flux_spacetime_variables_tag = ::Tags::Variables<
      tmpl::list<gr::Tags::SqrtDetSpatialMetric<>, gr::Tags::SpatialMetric<3>>>;
  using inverse_spatial_metric_tag = gr::Tags::InverseSpatialMetric<3>;
};

template <bool mesh_is_moving>
domain::creators::Brick create_a_brick(const size_t num_dg_pts,
                                       const double initial_time) {
  auto time_dependence_ptr = [&]() {
    if constexpr (mesh_is_moving) {
      const std::array<double, 3> mesh_velocity{1, 2, 3};
      return std::make_unique<
          domain::creators::time_dependence::UniformTranslation<3>>(
          initial_time, mesh_velocity);
    } else {
      return nullptr;
    }
  }();
  const auto lower_bounds = make_array<3, double>(3.0);
  const auto upper_bounds = make_array<3, double>(5.0);
  const auto refinement_levels = make_array<3, size_t>(0);
  return domain::creators::Brick(lower_bounds, upper_bounds, refinement_levels,
                                 make_array<3, size_t>(num_dg_pts),
                                 make_array<3, bool>(true),
                                 std::move(time_dependence_ptr));
}

std::array<Mesh<3>, 3> create_face_centered_meshes(
    const Mesh<3> cell_centered_mesh) {
  std::array<Mesh<3>, 3> result{};
  for (size_t dim = 0; dim < 3; ++dim) {
    const auto basis = make_array<3>(cell_centered_mesh.basis(0));
    auto quadrature = make_array<3>(cell_centered_mesh.quadrature(0));
    auto extents = make_array<3>(cell_centered_mesh.extents(0));
    gsl::at(extents, dim) = cell_centered_mesh.extents(0) + 1;
    gsl::at(quadrature, dim) = Spectral::Quadrature::FaceCentered;
    const Mesh<3> face_centered_mesh{extents, basis, quadrature};
    gsl::at(result, dim) = Mesh<3>{extents, basis, quadrature};
  }
  return result;
}

template <bool test_for_moving_mesh, InitialDataType initial_data_type>
void test(const gsl::not_null<std::mt19937*> gen) {
  //
  //
  //
  //
  //

  const double initial_time = 0.5;
  std::uniform_real_distribution<> distribution_time(1.0, 2.0);
  const double random_time{
      // make the random time strictly different from the initial time
      make_with_random_values<double>(gen, make_not_null(&distribution_time))};

  // Create a 3D element [3.0, 5.0]^3  for the test
  const size_t num_dg_pts = 3;
  const auto brick = [&]() {
    if constexpr (test_for_moving_mesh) {
      return create_a_brick<true>(num_dg_pts, initial_time);
    } else {
      return create_a_brick<false>(num_dg_pts, initial_time);
    }
  }();
  const auto domain = brick.create_domain();
  const auto element_id = ElementId<3>{0};

  const auto& block = domain.blocks()[element_id.block_id()];
  Element<3> element = domain::Initialization::create_initial_element(
      element_id, domain.blocks().at(0),
      std::vector<std::array<size_t, 3>>{{0, 0, 0}});

  const auto element_map = ElementMap<3, Frame::Grid>{
      element_id, block.is_time_dependent()
                      ? block.moving_mesh_logical_to_grid_map().get_clone()
                      : block.stationary_map().get_to_grid_frame()};

  const Mesh<3> dg_mesh{num_dg_pts, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh<3>(dg_mesh);
  const auto face_centered_meshes = create_face_centered_meshes(subcell_mesh);

  std::unique_ptr<::domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>>
      grid_to_inertial_map;
  if (block.is_time_dependent()) {
    grid_to_inertial_map = block.moving_mesh_grid_to_inertial_map().get_clone();
  } else {
    grid_to_inertial_map =
        ::domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            ::domain::CoordinateMaps::Identity<3>{});
  }

  const auto compute_inertial_coords = [&](const Mesh<3> mesh,
                                           const double time) {
    return (*grid_to_inertial_map)(element_map(logical_coordinates(mesh)), time,
                                   brick.functions_of_time());
  };

  const auto initial_subcell_inertial_coords =
      compute_inertial_coords(subcell_mesh, initial_time);
  const auto subcell_inertial_coords =
      compute_inertial_coords(subcell_mesh, random_time);

  std::array<tnsr::I<DataVector, 3, Frame::Inertial>, 3>
      initial_face_centered_inertial_coords_array{};
  std::array<tnsr::I<DataVector, 3, Frame::Inertial>, 3>
      face_centered_inertial_coords_array{};
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(initial_face_centered_inertial_coords_array, i) =
        compute_inertial_coords(gsl::at(face_centered_meshes, i), initial_time);
    gsl::at(face_centered_inertial_coords_array, i) =
        compute_inertial_coords(gsl::at(face_centered_meshes, i), random_time);
  }

  using gr_variables_tag =
      ::Tags::Variables<SystemForTest::spacetime_variables_tag::tags_list>;
  using subcell_gr_variables_tag =
      evolution::dg::subcell::Tags::Inactive<gr_variables_tag>;
  using subcell_face_gr_variables_tag =
      evolution::dg::subcell::Tags::OnSubcellFaces<
          typename SystemForTest::flux_spacetime_variables_tag, 3>;

  const auto solution = []() {
    if constexpr (initial_data_type == InitialDataType::CompileTime) {
      return gr::Solutions::KerrSchild{1.0, make_array<3, double>(0.0),
                                       make_array<3, double>(0.0)};
    }
    if constexpr (initial_data_type == InitialDataType::Runtime) {
      return RelativisticEuler::Solutions::TovStar{
          1.0e-3,
          EquationsOfState::PolytropicFluid<true>{100.0, 2.0}.get_clone(),
          RelativisticEuler::Solutions::TovCoordinates::Schwarzschild};
    }
  }();

  auto box = [&]() {
    // Since we want to test that the BackgroundGrVars action properly
    // initializes (allocate + assign) the background GR variables on
    // cell-centered and face-centered coordinates, use an empty Variables
    // objects here for creating a box.
    if constexpr (initial_data_type == InitialDataType::CompileTime) {
      return db::create<db::AddSimpleTags<
          ::Tags::Time, domain::Tags::Domain<3>, domain::Tags::Element<3>,
          domain::Tags::ElementMap<3, Frame::Grid>,
          domain::CoordinateMaps::Tags::CoordinateMap<3, Frame::Grid,
                                                      Frame::Inertial>,
          domain::Tags::FunctionsOfTimeInitialize,
          evolution::dg::subcell::Tags::Mesh<3>,
          evolution::dg::subcell::Tags::Coordinates<3, Frame::Inertial>,
          subcell_gr_variables_tag, subcell_face_gr_variables_tag,
          ::Tags::AnalyticSolution<gr::Solutions::KerrSchild>>>(
          initial_time, brick.create_domain(), element,
          ElementMap<3, Frame::Grid>{
              element_id,
              block.is_time_dependent()
                  ? block.moving_mesh_logical_to_grid_map().get_clone()
                  : block.stationary_map().get_to_grid_frame()},
          std::move(grid_to_inertial_map),
          clone_unique_ptrs(brick.functions_of_time()), subcell_mesh,
          initial_subcell_inertial_coords,
          typename subcell_gr_variables_tag::type{},
          typename subcell_face_gr_variables_tag::type{}, solution);
    }
    if constexpr (initial_data_type == InitialDataType::Runtime) {
      return db::create<db::AddSimpleTags<
          ::Tags::Time, domain::Tags::Domain<3>, domain::Tags::Element<3>,
          domain::Tags::ElementMap<3, Frame::Grid>,
          domain::CoordinateMaps::Tags::CoordinateMap<3, Frame::Grid,
                                                      Frame::Inertial>,
          domain::Tags::FunctionsOfTimeInitialize,
          evolution::dg::subcell::Tags::Mesh<3>,
          evolution::dg::subcell::Tags::Coordinates<3, Frame::Inertial>,
          subcell_gr_variables_tag, subcell_face_gr_variables_tag,
          evolution::initial_data::Tags::InitialData>>(
          initial_time, brick.create_domain(), element,
          ElementMap<3, Frame::Grid>{
              element_id,
              block.is_time_dependent()
                  ? block.moving_mesh_logical_to_grid_map().get_clone()
                  : block.stationary_map().get_to_grid_frame()},
          std::move(grid_to_inertial_map),
          clone_unique_ptrs(brick.functions_of_time()), subcell_mesh,
          initial_subcell_inertial_coords,
          typename subcell_gr_variables_tag::type{},
          typename subcell_face_gr_variables_tag::type{}, solution.get_clone());
    }
  }();

  // Execute the action, and check that it has put correct values of GR
  // variables in the box.
  evolution::dg::subcell::Actions::BackgroundGrVars<SystemForTest, false>::
      apply(box, tuples::TaggedTuple<>{},
            Parallel::GlobalCache<MetavariablesForTest>{}, 0, tmpl::list<>{},
            std::unique_ptr<size_t>{}.get());

  // check cell-centered values
  const auto expected_initial_cell_centered_gr_vars =
      solution.variables(initial_subcell_inertial_coords, initial_time,
                         gr_variables_tag::tags_list{});
  tmpl::for_each<gr_variables_tag::tags_list>([&](const auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    CHECK_ITERABLE_APPROX(get<evolution::dg::subcell::Tags::Inactive<tag>>(
                              get<subcell_gr_variables_tag>(box)),
                          get<tag>(expected_initial_cell_centered_gr_vars));
  });

  for (size_t d = 0; d < 3; ++d) {
    // check face-centered values
    const auto expected_face_centered_vars = solution.variables(
        gsl::at(initial_face_centered_inertial_coords_array, d), initial_time,
        subcell_face_gr_variables_tag::tag::tags_list{});

    tmpl::for_each<subcell_face_gr_variables_tag::tag::tags_list>(
        [&](const auto tag_v) {
          using tag = tmpl::type_from<decltype(tag_v)>;
          CHECK_ITERABLE_APPROX(
              get<tag>(expected_face_centered_vars),
              get<tag>(gsl::at(get<subcell_face_gr_variables_tag>(box), d)));
        });
  }

  // Mutate time and inertial coords to those at t = `random_time` and apply the
  // action again. Then check that the action evaluated correct values of GR
  // variables at a later time.
  db::mutate<::Tags::Time,
             evolution::dg::subcell::Tags::Coordinates<3, Frame::Inertial>>(
      make_not_null(&box),
      [&random_time, &subcell_inertial_coords](const auto time_ptr,
                                               const auto inertial_coords_ptr) {
        *time_ptr = random_time;
        *inertial_coords_ptr = subcell_inertial_coords;
      });

  evolution::dg::subcell::Actions::BackgroundGrVars<SystemForTest, false>::
      apply(box, tuples::TaggedTuple<>{},
            Parallel::GlobalCache<MetavariablesForTest>{}, 0, tmpl::list<>{},
            std::unique_ptr<size_t>{}.get());

  if constexpr (test_for_moving_mesh) {
    const auto expected_cell_centered_gr_vars = solution.variables(
        subcell_inertial_coords, random_time, gr_variables_tag::tags_list{});

    // check cell-centered
    tmpl::for_each<gr_variables_tag::tags_list>([&](const auto tag_v) {
      using tag = tmpl::type_from<decltype(tag_v)>;
      CHECK_ITERABLE_APPROX(get<evolution::dg::subcell::Tags::Inactive<tag>>(
                                get<subcell_gr_variables_tag>(box)),
                            get<tag>(expected_cell_centered_gr_vars));
    });

    // check face-centered values
    for (size_t d = 0; d < 3; ++d) {
      const auto expected_face_centered_vars = solution.variables(
          gsl::at(face_centered_inertial_coords_array, d), random_time,
          subcell_face_gr_variables_tag::tag::tags_list{});
      tmpl::for_each<subcell_face_gr_variables_tag::tag::tags_list>(
          [&](const auto tag_v) {
            using tag = tmpl::type_from<decltype(tag_v)>;
            CHECK_ITERABLE_APPROX(
                get<tag>(expected_face_centered_vars),
                get<tag>(gsl::at(get<subcell_face_gr_variables_tag>(box), d)));
          });
    }
  } else {
    // check cell-centered values
    tmpl::for_each<gr_variables_tag::tags_list>([&](const auto tag_v) {
      using tag = tmpl::type_from<decltype(tag_v)>;
      CHECK_ITERABLE_APPROX(get<evolution::dg::subcell::Tags::Inactive<tag>>(
                                get<subcell_gr_variables_tag>(box)),
                            get<tag>(expected_initial_cell_centered_gr_vars));
    });

    // check face-centered values
    for (size_t d = 0; d < 3; ++d) {
      const auto expected_initial_face_centered_vars = solution.variables(
          gsl::at(initial_face_centered_inertial_coords_array, d), initial_time,
          subcell_face_gr_variables_tag::tag::tags_list{});

      tmpl::for_each<subcell_face_gr_variables_tag::tag::tags_list>(
          [&](const auto tag_v) {
            using tag = tmpl::type_from<decltype(tag_v)>;
            CHECK_ITERABLE_APPROX(
                get<tag>(expected_initial_face_centered_vars),
                get<tag>(gsl::at(get<subcell_face_gr_variables_tag>(box), d)));
          });
    }
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.Actions.BackgroundGrVars",
                  "[Unit][Evolution]") {
  MAKE_GENERATOR(gen);

  test<false, InitialDataType::CompileTime>(make_not_null(&gen));
  test<false, InitialDataType::Runtime>(make_not_null(&gen));

  test<true, InitialDataType::CompileTime>(make_not_null(&gen));
  test<true, InitialDataType::Runtime>(make_not_null(&gen));
}

}  // namespace
