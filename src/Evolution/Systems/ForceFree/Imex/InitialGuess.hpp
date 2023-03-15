// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Imex/GuessResult.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "Utilities/TMPL.hpp"

// temporary header files
#include <iostream>
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Imex/Tags/ImplicitHistory.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "Parallel/AlgorithmExecution.hpp"
// #include "Parallel/GlobalCache.hpp"
#include "Parallel/Printf.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags/Time.hpp"

#include "Domain/Tags.hpp"

#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"

#include <iostream>

// #include "Evolution/Systems/ForceFree/System.hpp"

/// \cond
class DataVector;
template <typename>
class Variables;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace ForceFree {

namespace Imex {

struct InitialGuess {
  using return_tags = tmpl::list<Tags::TildeE>;
  using argument_tags = tmpl::list<Tags::TildeB, Tags::ParallelConductivity,
                                   gr::Tags::Lapse<DataVector>,
                                   gr::Tags::SpatialMetric<DataVector, 3>>;
  static imex::GuessResult apply(
      gsl::not_null<tnsr::I<DataVector, 3>*> tilde_e,
      const tnsr::I<DataVector, 3>& tilde_b, const double parallel_conductivity,
      const Scalar<DataVector>& lapse,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
      const Variables<tmpl::list<Tags::TildeE>>& inhomogeneous_terms,
      double implicit_weight);
};

namespace Actions {
// template <bool before_imex>
// struct ObserveEdotB {
//   template <typename DbTags, typename... InboxTags, typename Metavariables,
//             typename ArrayIndex, typename ActionList,
//             typename ParallelComponent>
//   static Parallel::iterable_action_return_t apply(
//       db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>&
//       /*inboxes*/, const Parallel::GlobalCache<Metavariables>& /*cache*/,
//       const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
//       const ParallelComponent* const /*meta*/) {
//     const auto& tilde_e = get<Tags::TildeE>(box);
//     const auto& tilde_b = get<Tags::TildeB>(box);
//     const auto& lapse = get<gr::Tags::Lapse<DataVector>>(box);
//     const auto& spatial_metric =
//         get<gr::Tags::SpatialMetric<DataVector, 3>>(box);

//     // const auto& e_dot_b = get<Tags::ElectricFieldDotMagneticField>(box);

//     Variables<tmpl::list<::Tags::TempScalar<0>>> buffer{get(lapse).size()};
//     auto& tilde_e_dot_tilde_b = get<::Tags::TempScalar<0>>(buffer);
//     dot_product(make_not_null(&tilde_e_dot_tilde_b), tilde_e, tilde_b,
//                 spatial_metric);

//     // if (max(abs(get(tilde_e_dot_tilde_b))) > 1.0e-10) {
//     if constexpr (before_imex) {
//       // std::cout << "\n *
//       // ======================================================"
//       //              "==================  \n ---- Before Imex : "
//       //           << std::endl;
//       std::cout << "-------- " << std::endl;
//     } else {
//       std::cout << "\n ---- After Imex : " << std::endl;
//       // std::cout << "-------- " << std::endl;
//     }

//     // std::cout << " * TildeB = " << tilde_b << std::endl;
//     // std::cout << " * TildeE = " << tilde_e << std::endl;
//     // std::cout << " * TildeE = " << tilde_e.get(0)[0] << std::endl;
//     // std::cout << " * TildeB = " << tilde_b.get(1)[0] << std::endl;
//     // std::cout << " * TildeEx = " << tilde_e.get(0) << std::endl;
//     // std::cout << " * TildeBy = " << tilde_b.get(2) << std::endl;
//     // std::cout << " * E dot B 1 = " << get(e_dot_b)[0] << std::endl;
//     // std::cout << " * E dot B 2 = " << get(tilde_e_dot_tilde_b)[0] <<
//     // std::endl; std::cout << " * E dot B = " <<
//     // max(abs(get(tilde_e_dot_tilde_b)))
//     // << std::endl;
//     // }

//     return {Parallel::AlgorithmExecution::Continue, std::nullopt};
//   }
// };

// template <bool use_dg_subcell>
// struct OverwriteFields {
//   template <typename DbTags, typename... InboxTags, typename Metavariables,
//             typename ArrayIndex, typename ActionList,
//             typename ParallelComponent>
//   static Parallel::iterable_action_return_t apply(
//       db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>&
//       /*inboxes*/, const Parallel::GlobalCache<Metavariables>& /*cache*/,
//       const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
//       const ParallelComponent* const /*meta*/) {
//     // const auto& coords =
//     // get<domain::Tags::Coordinates<3, Frame::Inertial>>(box);

//     const auto& coords = [&box]() {
//       if constexpr (use_dg_subcell) {
//         const auto& active_grid =
//             get<evolution::dg::subcell::Tags::ActiveGrid>(box);
//         if (active_grid == evolution::dg::subcell::ActiveGrid::Dg) {
//           return get<domain::Tags::Coordinates<3, Frame::Inertial>>(box);
//         } else {
//           return get<
//               evolution::dg::subcell::Tags::Coordinates<3, Frame::Inertial>>(
//               box);
//         }
//       } else {
//         return get<domain::Tags::Coordinates<3, Frame::Inertial>>(box);
//       }
//     }();

//     const auto& velocity_field =
//         get<ForceFree::Tags::NsInteriorSpatialVelocity>(box);

//     ASSERT(velocity_field.get(0).size() == coords.get(0).size(), "hoho");

//     const auto& mask = get<ForceFree::Tags::NsInteriorMask>(box);

//     db::mutate<ForceFree::Tags::TildeB, ForceFree::Tags::TildeE,
//                ForceFree::Tags::TildePsi, ForceFree::Tags::TildePhi,
//                ForceFree::Tags::TildeQ>(
//         [&coords, &mask, &velocity_field](
//             const gsl::not_null<tnsr::I<DataVector, 3>*> tilde_b_ptr,
//             const gsl::not_null<tnsr::I<DataVector, 3>*> tilde_e_ptr,
//             const gsl::not_null<Scalar<DataVector>*> tilde_psi_ptr,
//             const gsl::not_null<Scalar<DataVector>*> tilde_phi_ptr,
//             const gsl::not_null<Scalar<DataVector>*> tilde_q_ptr) {
//           for (size_t i = 0; i < coords.get(0).size(); ++i) {
//             // const auto& xi = coords.get(0)[i];
//             // const auto& yi = coords.get(1)[i];
//             // const auto& zi = coords.get(2)[i];

//             // r = sqrt(square(xi) + square(yi) + square(zi));

//             const auto& bx = tilde_b_ptr->get(0)[i];
//             const auto& by = tilde_b_ptr->get(1)[i];
//             const auto& bz = tilde_b_ptr->get(2)[i];
//             const auto& vx = get<0>(velocity_field)[i];
//             const auto& vy = get<1>(velocity_field)[i];
//             const auto& vz = get<2>(velocity_field)[i];

//             if (get(mask)[i] < 0.0) {
//               tilde_e_ptr->get(0)[i] = by * vz - bz * vy;
//               tilde_e_ptr->get(1)[i] = bz * vx - bx * vz;
//               tilde_e_ptr->get(2)[i] = bx * vy - by * vx;
//               get(*tilde_psi_ptr)[i] = 0.0;
//               get(*tilde_q_ptr)[i] = 0.0;
//             }
//           }
//         },
//         make_not_null(&box));

//     return {Parallel::AlgorithmExecution::Continue, std::nullopt};
//   }
// };

// struct ComputeImplicitTildeJ {
//   template <typename DbTags, typename... InboxTags, typename Metavariables,
//             typename ArrayIndex, typename ActionList,
//             typename ParallelComponent>
//   static Parallel::iterable_action_return_t apply(
//    db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
//       const Parallel::GlobalCache<Metavariables>& /*cache*/,
//       const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
//       const ParallelComponent* const /*meta*/) {
// const double implicit_weight =
//     db::get<::Tags::TimeStepper<>>(box).implicit_weight(
//         make_not_null(
//             &db::get_mutable_reference<imex::Tags::ImplicitHistory<
//        ForceFree::System::ImplicitSector>>(make_not_null(&box))),
//         db::get<::Tags::TimeStep>(box));

// std::cout << " Implicit weight = " << implicit_weight << std::endl;

// if (implicit_weight > 0.0) {
//   db::mutate<ForceFree::Tags::TildeE,
//   ForceFree::Tags::IntermediateTildeE,
//              ForceFree::Tags::ImplicitJ>(
//       make_not_null(&box),
//       [&implicit_weight](
//           const gsl::not_null<tnsr::I<DataVector, 3>*> tilde_e_ptr,
//           const gsl::not_null<tnsr::I<DataVector, 3>*>
//               intermediate_tilde_e_ptr,
//           const gsl::not_null<tnsr::I<DataVector, 3>*>
//               implicit_tilde_j_ptr) {
//         for (size_t i = 0; i < 3; ++i) {
//           (*implicit_tilde_j_ptr).get(i) =
//               ((*intermediate_tilde_e_ptr).get(i) -
//               (*tilde_e_ptr).get(i)) / implicit_weight;
//         }
//       });
// } else {
// }

// return {Parallel::AlgorithmExecution::Continue, std::nullopt};
// }
// };

}  // namespace Actions

struct AnalyticSolution {
  using return_tags = tmpl::list<Tags::TildeE>;
  using argument_tags = tmpl::list<Tags::TildeB, Tags::ParallelConductivity,
                                   gr::Tags::Lapse<DataVector>,
                                   gr::Tags::SpatialMetric<DataVector, 3>>;
  static imex::GuessResult apply(
      const gsl::not_null<tnsr::I<DataVector, 3>*> tilde_e,
      const tnsr::I<DataVector, 3>& tilde_b, const double parallel_conductivity,
      const Scalar<DataVector>& lapse,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
      const Variables<tmpl::list<Tags::TildeE>>& inhomogeneous_terms,
      const double implicit_weight) {
    // Solution for source terms
    // S[v2^ij] = v3^i v3^j - nt v2^ij
    // S[v3^i] = -v1 v3^i

    Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
                         ::Tags::TempScalar<2>>>
        buffer{get(lapse).size()};
    auto& tilde_b_squared = get<::Tags::TempScalar<0>>(buffer);
    auto& tilde_e_dot_tilde_b = get<::Tags::TempScalar<1>>(buffer);
    dot_product(make_not_null(&tilde_b_squared), tilde_b, tilde_b,
                spatial_metric);
    dot_product(make_not_null(&tilde_e_dot_tilde_b),
                get<Tags::TildeE>(inhomogeneous_terms), tilde_b,
                spatial_metric);

    auto& common_factor = get<::Tags::TempScalar<2>>(buffer);
    get(common_factor) = implicit_weight * parallel_conductivity * get(lapse);

    // Solving  v3^i = X - w v1 v3^i  gives  v3^i = X / (1 + w v1)
    for (size_t i = 0; i < 3; ++i) {
      (*tilde_e).get(i) =
          get<Tags::TildeE>(inhomogeneous_terms).get(i) -
          get(common_factor) * get(tilde_e_dot_tilde_b) * tilde_b.get(i) /
              ((1.0 + get(common_factor)) * get(tilde_b_squared));
    }

    return imex::GuessResult::ExactSolution;
  }
};

}  // namespace Imex

}  // namespace ForceFree
