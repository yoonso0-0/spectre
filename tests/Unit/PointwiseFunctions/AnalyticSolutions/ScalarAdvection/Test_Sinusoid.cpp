// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/AnalyticSolutions/ScalarAdvection/Sinusoid.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

void test_create() noexcept {
  const auto sine_wave =
      TestHelpers::test_creation<ScalarAdvection::Solutions::Sinusoid>("");
  CHECK(sine_wave == ScalarAdvection::Solutions::Sinusoid());
}

void test_serialize() noexcept {
  ScalarAdvection::Solutions::Sinusoid sine_wave;
  test_serialization(sine_wave);
}

void test_move() noexcept {
  ScalarAdvection::Solutions::Sinusoid sine_wave;
  ScalarAdvection::Solutions::Sinusoid sine_wave_copy;
  test_move_semantics(std::move(sine_wave), sine_wave_copy);  //  NOLINT
}

struct SinusoidProxy : ScalarAdvection::Solutions::Sinusoid {
  using ScalarAdvection::Solutions::Sinusoid::Sinusoid;

  using variables_tags = tmpl::list<ScalarAdvection::Tags::U>;

  template <typename DataType>
  tuples::tagged_tuple_from_typelist<variables_tags> retrieve_variables(
      const tnsr::I<DataType, 1>& x, double t) const noexcept {
    return this->variables(x, t, variables_tags{});
  }
};

void verify_solution(const gsl::not_null<std::mt19937*> generator,
                     const size_t number_of_pts) {
  // DataVectors to store values
  Scalar<DataVector> u_sol{number_of_pts};
  Scalar<DataVector> u_test{number_of_pts};

  // generate random 1D grid points
  std::uniform_real_distribution<> distribution_coords(-1.0, 1.0);
  const auto x = make_with_random_values<tnsr::I<DataVector, 1>>(
      generator, make_not_null(&distribution_coords), u_sol);

  // test for random time
  std::uniform_real_distribution<> distribution_time(0.0, 10.0);
  SinusoidProxy sine_wave;
  for (size_t i = 0; i < 3; ++i) {
    double t{make_with_random_values<double>(
        generator, make_not_null(&distribution_time))};

    u_sol = get<ScalarAdvection::Tags::U>(sine_wave.retrieve_variables(x, t));
    u_test = pypp::call<Scalar<DataVector>>("Sinusoid", "u", x, t);
    CHECK_ITERABLE_APPROX(u_sol, u_test);
  }
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.ScalarAdvection.Sinusoid",
    "[Unit][PointwiseFunctions]") {
  test_create();
  test_serialize();
  test_move();

  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/ScalarAdvection"};
  MAKE_GENERATOR(gen);
  verify_solution(make_not_null(&gen), 10);
}
