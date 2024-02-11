// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <memory>
#include <pup.h>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "Options/Options.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace ForceFree::Solutions {

/*!
 * \brief The magnetosphere of an isolated rotating star with dipolar initial
 * magnetic field in the flat spacetime. This is a toy model of a pulsar
 * magnetosphere.
 *
 */
class RotatingDipoleBoundary : public evolution::initial_data::InitialData,
                               public MarkAsAnalyticSolution {
 public:
  struct VectorPotentialAmplitude {
    using type = double;
    static constexpr Options::String help = {
        "The vector potential amplitude A_0"};
  };

  struct Varpi0 {
    using type = double;
    static constexpr Options::String help = {"The length constant varpi_0"};
    static type lower_bound() { return 0.0; }
  };

  struct Delta {
    using type = double;
    static constexpr Options::String help = {
        "A small value used to regularize magnetic fields at r=0."};
    static type lower_bound() { return 0.0; }
  };

  struct AngularVelocity {
    using type = double;
    static constexpr Options::String help = {
        "Rotation angular velocity of the star."};
    static type upper_bound() { return 1.0; }
    static type lower_bound() { return -1.0; }
  };

  struct TiltAngle {
    using type = double;
    static constexpr Options::String help = {
        "Angle between the rotation axis (z) and magnetic axis at t = 0."};
    static type upper_bound() { return M_PI; }
    static type lower_bound() { return 0.0; }
  };

  using options = tmpl::list<VectorPotentialAmplitude, Varpi0, Delta,
                             AngularVelocity, TiltAngle>;
  static constexpr Options::String help{
      "Magnetosphere of an isolated rotating star with dipole magnetic field."};

  RotatingDipoleBoundary() = default;
  RotatingDipoleBoundary(const RotatingDipoleBoundary&) = default;
  RotatingDipoleBoundary& operator=(const RotatingDipoleBoundary&) = default;
  RotatingDipoleBoundary(RotatingDipoleBoundary&&) = default;
  RotatingDipoleBoundary& operator=(RotatingDipoleBoundary&&) = default;
  ~RotatingDipoleBoundary() override = default;

  RotatingDipoleBoundary(double vector_potential_amplitude, double varpi0,
                         double delta, double angular_velocity,
                         double tilt_angle,
                         const Options::Context& context = {});

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override;

  /// \cond
  explicit RotatingDipoleBoundary(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(RotatingDipoleBoundary);
  /// \endcond

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

  /// @{
  /// Retrieve the EM variables.
  auto variables(const tnsr::I<DataVector, 3>& coords, double t,
                 tmpl::list<Tags::TildeE> /*meta*/) const
      -> tuples::TaggedTuple<Tags::TildeE>;

  auto variables(const tnsr::I<DataVector, 3>& coords, double t,
                 tmpl::list<Tags::TildeB> /*meta*/) const
      -> tuples::TaggedTuple<Tags::TildeB>;

  static auto variables(const tnsr::I<DataVector, 3>& coords, double t,
                        tmpl::list<Tags::TildePsi> /*meta*/)
      -> tuples::TaggedTuple<Tags::TildePsi>;

  static auto variables(const tnsr::I<DataVector, 3>& coords, double t,
                        tmpl::list<Tags::TildePhi> /*meta*/)
      -> tuples::TaggedTuple<Tags::TildePhi>;

  static auto variables(const tnsr::I<DataVector, 3>& coords, double t,
                        tmpl::list<Tags::TildeQ> /*meta*/)
      -> tuples::TaggedTuple<Tags::TildeQ>;
  /// @}

  /// Retrieve a collection of EM variables at position x
  template <typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataVector, 3>& x,
                                         const double t,
                                         tmpl::list<Tags...> /*meta*/) const {
    static_assert(sizeof...(Tags) > 1,
                  "The generic template will recurse infinitely if only one "
                  "tag is being retrieved.");
    return {get<Tags>(variables(x, t, tmpl::list<Tags>{}))...};
  }

  /// Retrieve the metric variables
  template <typename Tag>
  tuples::TaggedTuple<Tag> variables(const tnsr::I<DataVector, 3>& x, double t,
                                     tmpl::list<Tag> /*meta*/) const {
    return background_spacetime_.variables(x, t, tmpl::list<Tag>{});
  }

 private:
  double vector_potential_amplitude_ =
      std::numeric_limits<double>::signaling_NaN();
  double varpi0_ = std::numeric_limits<double>::signaling_NaN();
  double delta_ = std::numeric_limits<double>::signaling_NaN();
  double angular_velocity_ = std::numeric_limits<double>::signaling_NaN();
  double tilt_angle_ = std::numeric_limits<double>::signaling_NaN();
  gr::Solutions::Minkowski<3> background_spacetime_{};

  friend bool operator==(const RotatingDipoleBoundary& lhs,
                         const RotatingDipoleBoundary& rhs);
};

bool operator!=(const RotatingDipoleBoundary& lhs,
                const RotatingDipoleBoundary& rhs);

}  // namespace ForceFree::Solutions
