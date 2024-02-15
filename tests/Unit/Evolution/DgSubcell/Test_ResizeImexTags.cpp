// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/ResizeImexTags.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/Imex/GuessResult.hpp"
#include "Evolution/Imex/Protocols/ImexSystem.hpp"
#include "Evolution/Imex/Protocols/ImplicitSource.hpp"
#include "Evolution/Imex/Tags/ImplicitHistory.hpp"
#include "Evolution/Imex/Tags/SolveFailures.hpp"
#include "NumericalAlgorithms/FiniteDifference/DerivativeOrder.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Protocols/StaticReturnApplyable.hpp"
#include "Utilities/TMPL.hpp"

namespace {

struct Scalar1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct DummyInitialGuess {
  using return_tags = tmpl::list<Scalar1>;
  using argument_tags = tmpl::list<>;
  static std::vector<imex::GuessResult> apply(
      gsl::not_null<Scalar<DataVector>*> /*dummy_scalar*/) {
    return std::vector<imex::GuessResult>{};
  }
};

struct DummySource : tt::ConformsTo<imex::protocols::ImplicitSource>,
                     tt::ConformsTo<protocols::StaticReturnApplyable> {
  using return_tags = tmpl::list<::Tags::Source<Scalar1>>;
  using argument_tags = tmpl::list<>;
  static void apply(gsl::not_null<Scalar<DataVector>*> /*dummy_scalar*/) {}
};

struct DummyImplicitSector : tt::ConformsTo<imex::protocols::ImplicitSector> {
  using tensors = tmpl::list<Scalar1>;
  using initial_guess = DummyInitialGuess;

  struct DummySolve {
    using tags_from_evolution = tmpl::list<>;
    using simple_tags = tmpl::list<>;
    using compute_tags = tmpl::list<>;
    using source_prep = tmpl::list<>;
    using jacobian_prep = tmpl::list<>;

    using source = DummySource;
    using jacobian = imex::NoJacobianBecauseSolutionIsAnalytic;
  };

  using solve_attempts = tmpl::list<DummySolve>;
};

void test(const bool dg_to_fd) {
  const size_t num_dg_pts{5};
  const size_t num_fd_pts{10};
  const Mesh<1> dg_mesh{num_dg_pts, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<1> subcell_mesh{num_fd_pts, Spectral::Basis::FiniteDifference,
                             Spectral::Quadrature::CellCentered};

  const auto subcell_options = evolution::dg::subcell::SubcellOptions{
      4.0,
      1_st,
      1.0e-3,
      1.0e-4,
      false,
      false,
      evolution::dg::subcell::fd::ReconstructionMethod::DimByDim,
      false,
      std::nullopt,
      ::fd::DerivativeOrder::Two,
      1,
      1,
      1};

  const size_t num_grid_pts_before = dg_to_fd ? num_dg_pts : num_fd_pts;

  TimeSteppers::History<Variables<typename DummyImplicitSector::tensors>>
      implicit_history{1};
  const Slab slab{0.0, 1.0};
  const auto insert_history = [&](const TimeStepId& time_step_id) {
    implicit_history.insert(time_step_id,
                            Variables<typename DummyImplicitSector::tensors>{
                                num_grid_pts_before, 0.0},
                            typename decltype(implicit_history)::DerivVars{
                                num_grid_pts_before, 0.0});
  };
  insert_history(TimeStepId{true, 0, slab.start()});
  if (dg_to_fd) {
    insert_history(TimeStepId{true, 1, slab.end()});
  }
  const Scalar<DataVector> solve_failure{num_grid_pts_before, 0.0};

  auto box = db::create<
      db::AddSimpleTags<::imex::Tags::ImplicitHistory<DummyImplicitSector>,
                        ::imex::Tags::SolveFailures<DummyImplicitSector>>>(
      implicit_history, solve_failure);

  if (dg_to_fd) {
    evolution::dg::subcell::detail::ResizeImexTags::apply<
        true, tmpl::list<DummyImplicitSector>>(box, dg_mesh, subcell_mesh,
                                               subcell_options);
  } else {
    evolution::dg::subcell::detail::ResizeImexTags::apply<
        false, tmpl::list<DummyImplicitSector>>(box, dg_mesh, subcell_mesh,
                                                subcell_options);
  }

  const size_t expected_num_grid_pts = dg_to_fd ? num_fd_pts : num_dg_pts;
  const auto& resized_history =
      db::get<::imex::Tags::ImplicitHistory<DummyImplicitSector>>(box);
  REQUIRE(resized_history.size() == 1);
  REQUIRE(resized_history[0].value.has_value());
  CHECK(resized_history[0].value->number_of_grid_points() ==
        expected_num_grid_pts);
  CHECK(resized_history[0].derivative.number_of_grid_points() ==
        expected_num_grid_pts);
  CHECK(get(db::get<::imex::Tags::SolveFailures<DummyImplicitSector>>(box))
            .size() == expected_num_grid_pts);
}

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.ResizeImexTags",
                  "[Evolution][Unit]") {
  test(false);
  test(true);
}
}  // namespace
