// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <memory>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/FrameTransform.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "Options/Options.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/ForceFree/MagnetosphericWald.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/SphericalTorus.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/SphericalKerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/KerrSchildCoords.hpp"
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
 * \brief The magnetospheric Wald problem run with polar coordinates
 *
 *
 */
class PolarMagnetosphericWald
    : public virtual evolution::initial_data::InitialData,
      public MarkAsAnalyticData {
 public:
  struct TorusParameters {
    using type = grmhd::AnalyticData::SphericalTorus;
    static constexpr Options::String help =
        "Parameters for the evolution region.";
  };

  using options = tmpl::list<TorusParameters>;
  static constexpr Options::String help{"Magnetospheric Wald problem"};

  PolarMagnetosphericWald() = default;
  PolarMagnetosphericWald(const PolarMagnetosphericWald&) = default;
  PolarMagnetosphericWald& operator=(const PolarMagnetosphericWald&) = default;
  PolarMagnetosphericWald(PolarMagnetosphericWald&&) = default;
  PolarMagnetosphericWald& operator=(PolarMagnetosphericWald&&) = default;
  ~PolarMagnetosphericWald() override = default;

  PolarMagnetosphericWald(grmhd::AnalyticData::SphericalTorus torus_map);

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override;

  /// \cond
  explicit PolarMagnetosphericWald(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(PolarMagnetosphericWald);
  /// \endcond

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataType, 3>& x,
                                         tmpl::list<Tags...> /*meta*/) const {
    // In this function, we label the coordinates this solution works
    // in with Frame::BlockLogical, and the coordinates the wrapped
    // solution uses Frame::Inertial.  This means the input and output
    // have to be converted to the correct label.

    const tnsr::I<DataType, 3> observation_coordinates(torus_map_(x));

    using dependencies = tmpl::map<
        tmpl::pair<gr::AnalyticSolution<3>::DerivShift<DataType>,
                   gr::Tags::Shift<DataType, 3, Frame::Inertial>>,
        tmpl::pair<gr::AnalyticSolution<3>::DerivSpatialMetric<DataType>,
                   gr::Tags::SpatialMetric<DataType, 3, Frame::Inertial>>>;
    using required_tags = tmpl::remove_duplicates<
        tmpl::remove<tmpl::list<Tags..., tmpl::at<dependencies, Tags>...>,
                     tmpl::no_such_type_>>;

    auto observation_data = magnetospheric_wald_.variables(
        observation_coordinates, required_tags{});

    const auto jacobian = torus_map_.jacobian(x);
    const auto inv_jacobian = torus_map_.inv_jacobian(x);

    const auto change_frame = [this, &inv_jacobian, &jacobian, &x](
                                  const auto& data, auto tag) {
      using Tag = decltype(tag);
      auto result =
          transform::to_different_frame(get<Tag>(data), jacobian, inv_jacobian);

      if constexpr (std::is_same_v<
                        Tag, gr::AnalyticSolution<3>::DerivShift<DataType>>) {
        const auto deriv_inv_jacobian =
            torus_map_.derivative_of_inv_jacobian(x);
        const auto& shift =
            get<gr::Tags::Shift<DataType, 3, Frame::Inertial>>(data);
        for (size_t i = 0; i < 3; ++i) {
          for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 3; ++k) {
              result.get(i, j) +=
                  deriv_inv_jacobian.get(j, k, i) * shift.get(k);
            }
          }
        }
      } else if constexpr (std::is_same_v<
                               Tag, gr::AnalyticSolution<3>::DerivSpatialMetric<
                                        DataType>>) {
        const auto hessian = torus_map_.hessian(x);
        const auto& spatial_metric =
            get<gr::Tags::SpatialMetric<DataType, 3, Frame::Inertial>>(data);
        for (size_t i = 0; i < 3; ++i) {
          for (size_t j = 0; j < 3; ++j) {
            for (size_t k = j; k < 3; ++k) {
              for (size_t l = 0; l < 3; ++l) {
                for (size_t m = 0; m < 3; ++m) {
                  result.get(i, j, k) +=
                      (hessian.get(l, j, i) * jacobian.get(m, k) +
                       hessian.get(l, k, i) * jacobian.get(m, j)) *
                      spatial_metric.get(l, m);
                }
              }
            }
          }
        }
      } else if constexpr (std::is_same_v<
                               Tag, gr::Tags::SqrtDetSpatialMetric<DataType>>) {
        get(result) *= abs(get(determinant(jacobian)));
      }

      typename Tag::type result_with_replaced_frame{};
      std::copy(std::move_iterator(result.begin()),
                std::move_iterator(result.end()),
                result_with_replaced_frame.begin());
      return result_with_replaced_frame;
    };

    return {change_frame(observation_data, Tags{})...};
  }

 private:
  friend bool operator==(const PolarMagnetosphericWald& lhs,
                         const PolarMagnetosphericWald& rhs);
  double spin_ = 0.90;
  MagnetosphericWald magnetospheric_wald_{spin_, 0.0};
  grmhd::AnalyticData::SphericalTorus torus_map_;
};

bool operator!=(const PolarMagnetosphericWald& lhs,
                const PolarMagnetosphericWald& rhs);

}  // namespace ForceFree::AnalyticData
