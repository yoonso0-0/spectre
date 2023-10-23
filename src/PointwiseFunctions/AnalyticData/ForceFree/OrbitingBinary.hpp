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
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace ForceFree::AnalyticData {
/*!
 * \brief Orbiting binary
 *
 */
class OrbitingBinary : public evolution::initial_data::InitialData,
                       public MarkAsAnalyticData {
 public:
  struct AngularVelocityOne {
    using type = double;
    static constexpr Options::String help = {"Omega_1"};
    static type lower_bound() { return 0.0; }
  };

  struct AngularVelocityTwo {
    using type = double;
    static constexpr Options::String help = {"Omega_2"};
    static type lower_bound() { return 0.0; }
  };

  struct OrbitalRadius {
    using type = double;
    static constexpr Options::String help = {"Orbital radius"};
    static type lower_bound() { return 1.0; }
  };

  using options =
      tmpl::list<AngularVelocityOne, AngularVelocityTwo, OrbitalRadius>;

  static constexpr Options::String help{"Orbiting and spinning binary star"};

  OrbitingBinary() = default;
  OrbitingBinary(const OrbitingBinary&) = default;
  OrbitingBinary& operator=(const OrbitingBinary&) = default;
  OrbitingBinary(OrbitingBinary&&) = default;
  OrbitingBinary& operator=(OrbitingBinary&&) = default;
  ~OrbitingBinary() override = default;

  OrbitingBinary(double angular_velocity_one, double angular_velocity_two,
                 double orbital_radius, const Options::Context& context = {});

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override;

  /// \cond
  explicit OrbitingBinary(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(OrbitingBinary);
  /// \endcond

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

  /// @{
  /// Retrieve the EM variables.
  static auto variables(const tnsr::I<DataVector, 3>& coords,
                        tmpl::list<Tags::TildeE> /*meta*/)
      -> tuples::TaggedTuple<Tags::TildeE>;

  auto variables(const tnsr::I<DataVector, 3>& coords,
                 tmpl::list<Tags::TildeB> /*meta*/) const
      -> tuples::TaggedTuple<Tags::TildeB>;

  static auto variables(const tnsr::I<DataVector, 3>& coords,
                        tmpl::list<Tags::TildePsi> /*meta*/)
      -> tuples::TaggedTuple<Tags::TildePsi>;

  static auto variables(const tnsr::I<DataVector, 3>& coords,
                        tmpl::list<Tags::TildePhi> /*meta*/)
      -> tuples::TaggedTuple<Tags::TildePhi>;

  static auto variables(const tnsr::I<DataVector, 3>& coords,
                        tmpl::list<Tags::TildeQ> /*meta*/)
      -> tuples::TaggedTuple<Tags::TildeQ>;
  /// @}

  /// Retrieve a collection of EM variables at position x
  template <typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataVector, 3>& x,
                                         tmpl::list<Tags...> /*meta*/) const {
    static_assert(sizeof...(Tags) > 1,
                  "The generic template will recurse infinitely if only one "
                  "tag is being retrieved.");
    return {get<Tags>(variables(x, tmpl::list<Tags>{}))...};
  }

  /// Retrieve the metric variables
  template <typename Tag>
  tuples::TaggedTuple<Tag> variables(const tnsr::I<DataVector, 3>& x,
                                     tmpl::list<Tag> /*meta*/) const {
    constexpr double dummy_time = 0.0;
    return background_spacetime_.variables(x, dummy_time, tmpl::list<Tag>{});
  }

  // Returns the value of NS interior mask
  std::optional<Scalar<DataVector>> interior_mask(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x) const;
  std::optional<Scalar<DataVector>> interior_mask_one(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x) const;
  std::optional<Scalar<DataVector>> interior_mask_two(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x) const;

  // Returns the value of angular velocity.
  double angular_velocity_one() const { return angular_velocity_one_; };
  double angular_velocity_two() const { return angular_velocity_two_; };

  double orbital_radius() const { return orbital_radius_; };

 private:
  double angular_velocity_one_ = std::numeric_limits<double>::signaling_NaN();
  double angular_velocity_two_ = std::numeric_limits<double>::signaling_NaN();
  double orbital_radius_ = std::numeric_limits<double>::signaling_NaN();
  gr::Solutions::Minkowski<3> background_spacetime_{};

  friend bool operator==(const OrbitingBinary& lhs, const OrbitingBinary& rhs);
};

bool operator!=(const OrbitingBinary& lhs, const OrbitingBinary& rhs);

}  // namespace ForceFree::AnalyticData
