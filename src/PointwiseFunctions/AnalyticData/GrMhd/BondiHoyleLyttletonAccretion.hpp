// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <limits>

#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Context.hpp"
#include "Options/Options.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/SphericalKerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace grmhd::AnalyticData {

/*!
 * \brief Bondi-Hoyle-Lyttleton accretion problem in a fixed background black
 * hole spacetime.
 *
 * ....
 *
 */
class BondiHoyleLyttletonAccretion
    : public virtual evolution::initial_data::InitialData,
      public AnalyticDataBase,
      public hydro::TemperatureInitialization<BondiHoyleLyttletonAccretion>,
      public MarkAsAnalyticData {
 public:
  using equation_of_state_type = EquationsOfState::IdealFluid<true>;

  /// The mass of the Kerr black hole.
  struct Mass {
    using type = double;
    static constexpr Options::String help = {
        "The mass of the Kerr black hole."};
    static type lower_bound() { return 0.0; }
  };
  /// The [x,y,z] dimensionless spin \f$\vec{a}/M\f$ of the black hole.
  struct Spin {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "The [x,y,z] dimensionless spin of the black hole."};
  };

  /// The [x,y,z] coordinate center of the black hole.
  struct Center {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "The [x,y,z] coordinate center of the black hole."};
  };

  struct AdiabaticIndex {
    using type = double;
    static constexpr Options::String help = {
        "The adiabatic index of the fluid."};
  };

  struct SoundSpeed {
    using type = double;
    static constexpr Options::String help = {
        "The asymptotic sound speed of the fluid."};
  };

  struct WindVelocity {
    using type = double;
    static constexpr Options::String help = {
        "The asymptotic incoming velocity (from +x to -x) of the fluid."};
  };

  using options =
      tmpl::list<Mass, Spin, Center, AdiabaticIndex, SoundSpeed, WindVelocity>;

  static constexpr Options::String help = {
      "Bondi-Hoyle-Lyttleton accretion onto a Kerr black hole"};

  BondiHoyleLyttletonAccretion() = default;
  BondiHoyleLyttletonAccretion(const BondiHoyleLyttletonAccretion& /*rhs*/) =
      default;
  BondiHoyleLyttletonAccretion& operator=(
      const BondiHoyleLyttletonAccretion& /*rhs*/) = default;
  BondiHoyleLyttletonAccretion(BondiHoyleLyttletonAccretion&& /*rhs*/) =
      default;
  BondiHoyleLyttletonAccretion& operator=(
      BondiHoyleLyttletonAccretion&& /*rhs*/) = default;
  ~BondiHoyleLyttletonAccretion() override = default;

  BondiHoyleLyttletonAccretion(double mass, const std::array<double, 3>& spin,
                               const std::array<double, 3>& center,
                               double adiabatic_index, double sound_speed,
                               double wind_velocity,
                               const Options::Context& context = {});

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override;

  /// \cond
  explicit BondiHoyleLyttletonAccretion(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(BondiHoyleLyttletonAccretion);
  /// \endc

  /// @{
  /// Retrieve the GRMHD variables at a given position.
  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/)
      const -> tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::ElectronFraction<DataType>> /*meta*/)
      const -> tuples::TaggedTuple<hydro::Tags::ElectronFraction<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<hydro::Tags::Pressure<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::SpatialVelocity<DataType, 3>> /*meta*/)
      const -> tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DataType, 3>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::MagneticField<DataType, 3>> /*meta*/)
      const -> tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/)
      const -> tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>> /*meta*/)
      const -> tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::Temperature<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<hydro::Tags::Temperature<DataType>> {
    return TemperatureInitialization::variables(
        x, tmpl::list<hydro::Tags::Temperature<DataType>>{});
  }
  /// @}

  /// Retrieve a collection of hydrodynamic variables at position x
  template <typename DataType, typename Tag1, typename Tag2, typename... Tags>
  tuples::TaggedTuple<Tag1, Tag2, Tags...> variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<Tag1, Tag2, Tags...> /*meta*/) const {
    return {tuples::get<Tag1>(variables(x, tmpl::list<Tag1>{})),
            tuples::get<Tag2>(variables(x, tmpl::list<Tag2>{})),
            tuples::get<Tags>(variables(x, tmpl::list<Tags>{}))...};
  }

  /// Retrieve the metric variables
  template <typename DataType, typename Tag,
            Requires<tmpl::list_contains_v<
                gr::analytic_solution_tags<3, DataType>, Tag>> = nullptr>
  tuples::TaggedTuple<Tag> variables(const tnsr::I<DataType, 3>& x,
                                     tmpl::list<Tag> /*meta*/) const {
    constexpr double dummy_time = 0.0;
    return background_spacetime_.variables(x, dummy_time, tmpl::list<Tag>{});
  }

  const EquationsOfState::IdealFluid<true>& equation_of_state() const {
    return equation_of_state_;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) override;

 private:
  double adiabatic_index_ = std::numeric_limits<double>::signaling_NaN();
  double sound_speed_ = std::numeric_limits<double>::signaling_NaN();
  double wind_velocity_ = std::numeric_limits<double>::signaling_NaN();
  double pressure_ = std::numeric_limits<double>::signaling_NaN();
  EquationsOfState::IdealFluid<true> equation_of_state_{};
  gr::Solutions::SphericalKerrSchild background_spacetime_{};

  friend bool operator==(const BondiHoyleLyttletonAccretion& lhs,
                         const BondiHoyleLyttletonAccretion& rhs);
  friend bool operator!=(const BondiHoyleLyttletonAccretion& lhs,
                         const BondiHoyleLyttletonAccretion& rhs);
};

}  // namespace grmhd::AnalyticData
