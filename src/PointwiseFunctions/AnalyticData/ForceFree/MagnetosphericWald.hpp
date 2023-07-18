// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <memory>

#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/KerrSchildCoords.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/TMPL.hpp"

#include <iostream>

/// \cond
class DataVector;
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace ForceFree::AnalyticData {
/*!
 * \brief The magnetospheric Wald problem proposed in \cite Komissarov2004
 *
 * This is an initial value problem that evolves the magnetosphere of a rotating
 * black hole. The initial condition is given as same as the exact Wald solution
 * \cite Wald1974 (see also documentation of ForceFree::Solutions::ExactWald)
 *
 * \begin{equation}
 *  A_\mu = \frac{B_0}{2}(\phi_\mu + 2a t_\mu) ,
 * \end{equation}
 *
 * but electric field is set to zero at $t=0$.
 *
 * In the Kerr-Schild coordinates, densitized electric and magnetic fields are
 * given as
 *
 * \begin{equation}
 *  \tilde{E}^i = 0 ,
 * \end{equation}
 *
 * and
 *
 * \begin{align}
 *  \tilde{B}^x &= 2 a M B_0 \frac{r^2-a^2}{r^2+a^2}
 *                  \frac{r^3 z}{(r^4 + a^2z^2)^2}(ax - ry) , \\[1em]
 *  \tilde{B}^y &= 2 a M B_0 \frac{r^2-a^2}{r^2+a^2}
 *                  \frac{r^3 z}{(r^4 + a^2z^2)^2}(rx + ay) , \\[1em]
 *  \tilde{B}^z &= B_0
 *      \left[ 1 - \frac{2Ma^2r(r^2+z^2)}{(r^2+a^2)(r^4+a^2z^2)} \right] ,
 * \end{align}
 *
 * where $M$ and $a$ are mass and (dimensionless) spin of the Kerr black hole,
 * $B_0$ is the amplitude of magnetic field, and $r$ is the radial coordinate
 * defined in the Kerr-Schild coordinate system (see the documentation of
 * gr::Solutions::KerrSchild). We set $M=1$.
 *
 * There is no known exact solution to this problem, but numerical simulations
 * \cite Komissarov2004 \cite Paschalidis2013 \cite Etienne2017 report that it
 * converges to a steady state with a current sheet inside the black hole
 * ergosphere.
 *
 * TODO : add documentation of heck here..
 *
 */
class MagnetosphericWald : public evolution::initial_data::InitialData,
                           public MarkAsAnalyticData {
 public:
  struct Spin {
    using type = double;
    static constexpr Options::String help = {"The BH spin"};
    static type upper_bound() { return 1.0; }
    static type lower_bound() { return -1.0; }
  };

  using options = tmpl::list<Spin>;
  static constexpr Options::String help{"Magnetospheric Wald problem"};

  MagnetosphericWald() = default;
  MagnetosphericWald(const MagnetosphericWald&) = default;
  MagnetosphericWald& operator=(const MagnetosphericWald&) = default;
  MagnetosphericWald(MagnetosphericWald&&) = default;
  MagnetosphericWald& operator=(MagnetosphericWald&&) = default;
  ~MagnetosphericWald() override = default;

  MagnetosphericWald(double spin, const Options::Context& context = {});

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override;

  /// \cond
  explicit MagnetosphericWald(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(MagnetosphericWald);
  /// \endcond

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

  /// @{
  /// Retrieve the EM variables at (x,t).
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
    return background_spacetime_.variables(regularize_coords(x), dummy_time,
                                           tmpl::list<Tag>{});
  }

 private:
  double spin_ = std::numeric_limits<double>::signaling_NaN();
  gr::Solutions::KerrSchild background_spacetime_{};
  gr::KerrSchildCoords kerr_schild_coords_{};

  tnsr::I<DataVector, 3, Frame::Inertial> regularize_coords(
      const tnsr::I<DataVector, 3>& inertial_coords) const;

  friend bool operator==(const MagnetosphericWald& lhs,
                         const MagnetosphericWald& rhs);
  friend bool operator!=(const MagnetosphericWald& lhs,
                         const MagnetosphericWald& rhs);
};

}  // namespace ForceFree::AnalyticData
