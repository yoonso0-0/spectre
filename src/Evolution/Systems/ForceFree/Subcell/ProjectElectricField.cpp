// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/Subcell/ProjectElectricField.hpp"

#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/TMPL.hpp"

namespace ForceFree::subcell {

void project_electric_field_impl(
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_e,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const size_t number_of_grid_points) {
  Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
                       ::Tags::Tempi<0, 3>>>
      temp{number_of_grid_points};

  ASSERT(tilde_b.get(0).size() == number_of_grid_points,
         "ProjectElectricField impl : Grid size does not match. Should be "
             << number_of_grid_points << " but got " << tilde_b.get(0).size());

  auto& tilde_b_squared = get<::Tags::TempScalar<0>>(temp);
  auto& tilde_e_dot_tilde_b = get<::Tags::TempScalar<1>>(temp);

  dot_product(make_not_null(&tilde_b_squared), tilde_b, tilde_b,
              spatial_metric);
  dot_product(make_not_null(&tilde_e_dot_tilde_b), *tilde_e, tilde_b,
              spatial_metric);

  for (size_t d = 0; d < 3; ++d) {
    tilde_e->get(d) -=
        get(tilde_e_dot_tilde_b) * tilde_b.get(d) / get(tilde_b_squared);
  }
}

void ProjectElectricField::apply(
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_e,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const Mesh<3>& dg_mesh, const Mesh<3>& subcell_mesh,
    const bool did_rollback,
    const evolution::dg::subcell::ActiveGrid& active_grid) {
  if (did_rollback) {  // Active grid is FD
    project_electric_field_impl(tilde_e, tilde_b, spatial_metric,
                                subcell_mesh.number_of_grid_points());
  } else if (active_grid == evolution::dg::subcell::ActiveGrid::Dg) {
    project_electric_field_impl(tilde_e, tilde_b, spatial_metric,
                                dg_mesh.number_of_grid_points());
  }
}

}  // namespace ForceFree::subcell
