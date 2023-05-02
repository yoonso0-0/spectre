// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>

#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
// IWYU pragma: no_forward_declare Tensor
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace ForceFree::AnalyticData {

/*!
 * \brief Overlapping alfven waves from \cite Ripperda2021.
 *
 * Initial conditions are given as
 *
 * \begin{align}
 *  B^x & = \delta B \cos (k_\perp y - k_\parallel z) , \\
 *  B^y & = - \delta B \cos (k_\perp x + k_\parallel z) , \\
 *  B^z & = B_0
 * \end{align}
 *
 * and
 *
 * \begin{equation}
 *  E^i = (B^y, B^x, 0) .
 * \end{equation}
 *
 * We scale $B_0=1$ and specify $\delta B / B$ as an input value. Wavenumbers
 * are set $k_\perp = k_\parallel = 2\pi$.
 *
 * \note The original test \cite Ripperda2021 used $\delta B / B = 0.1$ and
 * $k_\perp L = k_\parallel L = 2\pi$.
 *
 * Nonlinearity develops after many wave-crossing times ($\approx (B/\delta
 * B)^2$), and waves become turbulent.
 *
 */
class OverlappingAlfvenWaves : public evolution::initial_data::InitialData,
                               public MarkAsAnalyticData {
 public:
  struct NormalizedWaveAmplitude {
    using type = double;
    static constexpr Options::String help = {"Delta B / B"};
    static type lower_bound() { return 0.0; }
  };

  using options = tmpl::list<NormalizedWaveAmplitude>;
  static constexpr Options::String help{"Alfvenic turbulence problem"};

  OverlappingAlfvenWaves() = default;
  OverlappingAlfvenWaves(const OverlappingAlfvenWaves&) = default;
  OverlappingAlfvenWaves& operator=(const OverlappingAlfvenWaves&) = default;
  OverlappingAlfvenWaves(OverlappingAlfvenWaves&&) = default;
  OverlappingAlfvenWaves& operator=(OverlappingAlfvenWaves&&) = default;
  ~OverlappingAlfvenWaves() override = default;

  OverlappingAlfvenWaves(double normalized_wave_amplitude,
                         const Options::Context& context = {});

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override;

  /// \cond
  explicit OverlappingAlfvenWaves(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(OverlappingAlfvenWaves);
  /// \endcond

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

  /// @{
  /// Retrieve the EM variables at (x,t).
  auto variables(const tnsr::I<DataVector, 3>& coords,
                 tmpl::list<Tags::TildeE> /*meta*/) const
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

  auto variables(const tnsr::I<DataVector, 3>& coords,
                 tmpl::list<Tags::TildeQ> /*meta*/) const
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

 private:
  double normalized_wave_amplitude_ =
      std::numeric_limits<double>::signaling_NaN();
  double k_perp_ = 2.0 * M_PI;
  double k_parallel_ = 2.0 * M_PI;
  gr::Solutions::Minkowski<3> background_spacetime_{};

  friend bool operator==(const OverlappingAlfvenWaves& lhs,
                         const OverlappingAlfvenWaves& rhs);
};

bool operator!=(const OverlappingAlfvenWaves& lhs,
                const OverlappingAlfvenWaves& rhs);

}  // namespace ForceFree::AnalyticData
