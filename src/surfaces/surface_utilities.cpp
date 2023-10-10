#include <ADS/app_namespaces.h>
#include <ADS/surface_utilities.h>

#include <libmesh/enum_preconditioner_type.h>
#include <libmesh/enum_solver_type.h>
#include <libmesh/explicit_system.h>
#include <libmesh/petsc_linear_solver.h>
#include <libmesh/petsc_matrix.h>

namespace ADS
{

void
update_jacobian(const std::string& J_sys_name, FEMeshPartitioner& fe_partitioner)
{
    EquationSystems* eq_sys = fe_partitioner.getEquationSystems();
    MeshBase& mesh = eq_sys->get_mesh();
    auto& J_sys = eq_sys->get_system<ExplicitSystem>(J_sys_name);
    DofMap& J_dof_map = J_sys.get_dof_map();
    J_dof_map.compute_sparsity(mesh);
    auto J_vec = dynamic_cast<libMesh::PetscVector<double>*>(J_sys.solution.get());
    std::unique_ptr<NumericVector<double>> F_c_vec(J_vec->zero_clone());
    auto F_vec = dynamic_cast<libMesh::PetscVector<double>*>(F_c_vec.get());

    auto& X_sys = eq_sys->get_system<System>(fe_partitioner.COORDINATES_SYSTEM_NAME);
    FEType X_fe_type = X_sys.get_dof_map().variable_type(0);
    NumericVector<double>* X_vec = X_sys.solution.get();
    auto X_petsc_vec = dynamic_cast<PetscVector<double>*>(X_vec);
    TBOX_ASSERT(X_petsc_vec != nullptr);
    const double* const X_local_soln = X_petsc_vec->get_array_read();
    FEDataManager::SystemDofMapCache& X_dof_map_cache =
        *fe_partitioner.getDofMapCache(fe_partitioner.COORDINATES_SYSTEM_NAME);

    std::vector<dof_id_type> J_dof_indices;

    std::unique_ptr<FEBase> fe = FEBase::build(mesh.mesh_dimension(), X_fe_type);
    std::unique_ptr<QBase> qrule = QBase::build(QGAUSS, mesh.mesh_dimension(), FIFTH);
    fe->attach_quadrature_rule(qrule.get());
    const std::vector<std::vector<double>>& phi = fe->get_phi();
    const std::vector<double>& JxW = fe->get_JxW();
    std::array<const std::vector<std::vector<double>>*, NDIM - 1> dphi_dxi;
    dphi_dxi[0] = &fe->get_dphidxi();
#if (NDIM == 3)
    dphi_dxi[1] = &fe->get_dphideta();
#endif

    std::unique_ptr<PetscLinearSolver<double>> solver(new PetscLinearSolver<double>(mesh.comm()));
    std::unique_ptr<PetscMatrix<double>> M_mat(new PetscMatrix<double>(mesh.comm()));
    M_mat->attach_dof_map(J_dof_map);
    M_mat->init();

    DenseMatrix<double> M_e;
    DenseVector<double> F_e;

    const MeshBase::const_element_iterator el_begin = mesh.active_elements_begin();
    const MeshBase::const_element_iterator el_end = mesh.active_elements_end();
    for (auto el_it = el_begin; el_it != el_end; ++el_it)
    {
        Elem* const elem = *el_it;
        J_dof_map.dof_indices(elem, J_dof_indices);
        const auto n_basis = static_cast<unsigned int>(J_dof_indices.size());
        M_e.resize(n_basis, n_basis);
        F_e.resize(n_basis);
        boost::multi_array<double, 2> x_node;

        const auto& X_dof_indices = X_dof_map_cache.dof_indices(elem);
        IBTK::get_values_for_interpolation(x_node, *X_petsc_vec, X_local_soln, X_dof_indices);
        boost::multi_array<double, 2> X_node(boost::extents[X_dof_indices[0].size()][X_dof_indices.size()]);
        for (unsigned int k = 0; k < elem->n_nodes(); ++k)
        {
            for (unsigned int d = 0; d < NDIM; ++d) X_node[k][d] = elem->point(k)(d);
        }
        fe->reinit(elem);
        VectorValue<double> X;
        F_e.zero();
        M_e.zero();
        for (unsigned int qp = 0; qp < phi[0].size(); ++qp)
        {
            X.zero();
            IBTK::interpolate(X, qp, x_node, phi);
            std::vector<VectorValue<double>> dx_dxi(2);
            std::vector<VectorValue<double>> dX_dxi(2);
            for (unsigned int l = 0; l < NDIM - 1; ++l)
            {
                IBTK::interpolate(dx_dxi[l], qp, x_node, *dphi_dxi[l]);
                IBTK::interpolate(dX_dxi[l], qp, X_node, *dphi_dxi[l]);
            }
#if (NDIM == 2)
            dx_dxi[1] = VectorValue<double>(0.0, 0.0, 1.0);
            dX_dxi[1] = VectorValue<double>(0.0, 0.0, 1.0);
#endif
            VectorValue<double> n = dx_dxi[0].cross(dx_dxi[1]);
            VectorValue<double> N = dX_dxi[0].cross(dX_dxi[1]);
            double J = N.norm() / n.norm();

            // We have Jacobian at quadrature point. Fill in system.
            for (unsigned int i = 0; i < n_basis; ++i)
            {
                for (unsigned int j = 0; j < n_basis; ++j)
                {
                    M_e(i, j) += (phi[i][qp] * phi[j][qp]) * JxW[qp];
                }
                F_e(i) += phi[i][qp] * J * JxW[qp];
            }
            J_dof_map.constrain_element_matrix_and_vector(M_e, F_e, J_dof_indices);
            M_mat->add_matrix(M_e, J_dof_indices);
            F_vec->add_vector(F_e, J_dof_indices);
        }
    }

    MatSetOption(M_mat->mat(), MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);
    MatSetOption(M_mat->mat(), MAT_SPD, PETSC_TRUE);
    MatSetOption(M_mat->mat(), MAT_SYMMETRY_ETERNAL, PETSC_TRUE);
    M_mat->close();

    solver->reuse_preconditioner(true);
    solver->set_preconditioner_type(JACOBI_PRECOND);
    solver->set_solver_type(MINRES);
    solver->init();

    solver->solve(*M_mat, *J_vec, *F_vec, 1.0e-8, 1000);
    KSPConvergedReason reason;
    int ierr = KSPGetConvergedReason(solver->ksp(), &reason);
    IBTK_CHKERRQ(ierr);
    bool converged = reason > 0;
    plog << "Projection converged: " << converged << "\n";

    X_petsc_vec->restore_array();
    J_vec->close();
    J_sys.update();
}
} // namespace ADS
