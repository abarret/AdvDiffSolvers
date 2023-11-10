#include "ADS/BoundaryReconstructCache.h"
#include "ADS/app_namespaces.h"
#include "ADS/ls_functions.h"
#include "ADS/reconstructions.h"

#include <ibtk/IndexUtilities.h>

namespace ADS
{
BoundaryReconstructCache::BoundaryReconstructCache(int ls_idx,
                                                   Pointer<PatchHierarchy<NDIM>> hierarchy,
                                                   std::shared_ptr<GeneralBoundaryMeshMapping> mesh_mapping,
                                                   const int stencil_size)
    : d_stencil_size(stencil_size), d_hierarchy(hierarchy), d_mesh_mapping(std::move(mesh_mapping)), d_ls_idx(ls_idx)
{
    // intentionally blank
}

BoundaryReconstructCache::BoundaryReconstructCache(const int stencil_size) : d_stencil_size(stencil_size)
{
    // intentionally blank
}

void
BoundaryReconstructCache::clearCache()
{
    d_weights_map.clear();
    d_update_weights = true;
}

void
BoundaryReconstructCache::setLSData(const int ls_idx)
{
    d_ls_idx = ls_idx;
    clearCache();
    d_update_weights = true;
}

void
BoundaryReconstructCache::setSign(bool use_positive)
{
    d_sign = use_positive ? 1.0 : -1.0;
    clearCache();
    d_update_weights = true;
}

void
BoundaryReconstructCache::setPatchHierarchy(Pointer<PatchHierarchy<NDIM>> hierarchy)
{
    d_hierarchy = hierarchy;
    clearCache();
    d_update_weights = true;
}

void
BoundaryReconstructCache::setMeshMapping(std::shared_ptr<GeneralBoundaryMeshMapping> mesh_mapping)
{
    d_mesh_mapping = std::move(mesh_mapping);
    clearCache();
    d_update_weights = true;
}

void
BoundaryReconstructCache::cacheData()
{
    // Loop through all parts of mesh mapping
    for (int part = 0; part < d_mesh_mapping->getNumParts(); ++part)
    {
        // Grab the position of the mesh
        const std::shared_ptr<FEMeshPartitioner>& mesh_partitioner = d_mesh_mapping->getMeshPartitioner(part);
        EquationSystems* eq_sys = mesh_partitioner->getEquationSystems();
        const System& X_sys = eq_sys->get_system(mesh_partitioner->COORDINATES_SYSTEM_NAME);
        NumericVector<double>* X_vec = X_sys.current_local_solution.get();
        const DofMap& X_dof_map = X_sys.get_dof_map();

        // Mesh should be on finest level.
        // TODO: Relax this constraint?
        const int ln = d_hierarchy->getFinestLevelNumber();
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        const std::vector<std::vector<Node*>>& patch_node_map = mesh_partitioner->getActivePatchNodeMap(ln);
        int patch_num = 0;
        for (PatchLevel<NDIM>::Iterator p(level); p; p++, ++patch_num)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
            const double* const dx = pgeom->getDx();
            const double* const xlow = pgeom->getXLower();
            const hier::Index<NDIM>& idx_low = patch->getBox().lower();
            Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(d_ls_idx);
            for (const auto& node : patch_node_map[patch_num])
            {
                // For each node, generate interp weights.
                const dof_id_type node_id = node->id();
                VectorNd x_node;
                for (int d = 0; d < NDIM; ++d)
                {
                    std::vector<dof_id_type> X_ind;
                    X_dof_map.dof_indices(node, X_ind, d);
                    x_node[d] = (*X_vec)(X_ind[0]);
                }

                // Determine the patch this element is in
                CellIndex<NDIM> idx_node = IndexUtilities::getCellIndex(x_node.data(), pgeom, patch->getBox());

                // Now flood fill to find group of cells to use for interpolation
                std::vector<CellIndex<NDIM>> test_idxs = { idx_node };
                std::vector<CellIndex<NDIM>> idx_list;
                std::vector<IBTK::VectorNd /*, Eigen::aligned_allocator<IBTK::VectorNd>*/> X_list;
                unsigned int i = 0;
                while (idx_list.size() < d_stencil_size)
                {
#ifndef NDEBUG
                    TBOX_ASSERT(i < test_idxs.size());
#endif
                    CellIndex<NDIM> test_idx = test_idxs[i];
                    if (node_to_cell(test_idx, *ls_data) * d_sign > 0.0)
                    {
                        idx_list.push_back(test_idx);
                        VectorNd x;
                        for (int d = 0; d < NDIM; ++d)
                            x[d] = xlow[d] + dx[d] * (static_cast<double>(test_idx(d) - idx_low(d)) + 0.5);
                        X_list.push_back(x);
                    }
                    // Now add neighboring indices if they haven't been visited.
                    IntVector<NDIM> l(-1, 0), r(1, 0), b(0, -1), u(0, 1);
                    CellIndex<NDIM> idx_l(test_idx + l), idx_r(test_idx + r);
                    CellIndex<NDIM> idx_u(test_idx + u), idx_b(test_idx + b);
                    if (std::find(test_idxs.begin(), test_idxs.end(), idx_l) == test_idxs.end())
                        test_idxs.push_back(idx_l);
                    if (std::find(test_idxs.begin(), test_idxs.end(), idx_r) == test_idxs.end())
                        test_idxs.push_back(idx_r);
                    if (std::find(test_idxs.begin(), test_idxs.end(), idx_u) == test_idxs.end())
                        test_idxs.push_back(idx_u);
                    if (std::find(test_idxs.begin(), test_idxs.end(), idx_b) == test_idxs.end())
                        test_idxs.push_back(idx_b);
                    ++i;
                }

                // Now that we have the patch indices, we can compute the weights.
                std::vector<double> wgts;
                auto L_rbf = [](const VectorNd& x, const VectorNd& y, void*) -> double { return (x - y).norm(); };
                auto L_polys =
                    [](const std::vector<VectorNd>& pt_vec, int poly_degree, double ds, const VectorNd& shft, void*)
                    -> MatrixXd { return PolynomialBasis::formMonomials(pt_vec, poly_degree, ds, shft).transpose(); };
                int poly_degree = 1;
                Reconstruct::RBFFD_reconstruct<VectorNd>(
                    wgts, x_node, X_list, poly_degree, dx, Reconstruct::rbf, L_rbf, nullptr, L_polys, nullptr);

                // Now store the weights.
                d_weights_map[std::make_pair(part, node_id)] = WeightStruct(patch, idx_list, wgts);
            }
        }
    }

    d_update_weights = false;
}

double
BoundaryReconstructCache::reconstruct(const int part, const int node_id, const int Q_idx)
{
    if (d_update_weights) cacheData();

    const WeightStruct& weight = getWeightStruct(part, node_id);
    Pointer<Patch<NDIM>> patch = weight.d_patch;
    Pointer<CellData<NDIM, double>> Q_data = patch->getPatchData(Q_idx);
    double val = 0.0;
    for (const auto& idx_wgt_pair : weight.d_idx_wgt_pair_vec)
    {
        val += idx_wgt_pair.second * (*Q_data)(idx_wgt_pair.first);
    }
    return val;
}

double
BoundaryReconstructCache::reconstruct(const int part, const int node_id, const int Q_idx) const
{
    const WeightStruct& weight = getWeightStruct(part, node_id);
    Pointer<Patch<NDIM>> patch = weight.d_patch;
    Pointer<CellData<NDIM, double>> Q_data = patch->getPatchData(Q_idx);
    double val = 0.0;
    for (const auto& idx_wgt_pair : weight.d_idx_wgt_pair_vec)
    {
        val += idx_wgt_pair.second * (*Q_data)(idx_wgt_pair.first);
    }
    return val;
}

void
BoundaryReconstructCache::reconstruct(const int part, const std::string& Q_str, const int Q_idx)
{
#ifndef NDEBUG
    TBOX_ASSERT(part < d_mesh_mapping->getNumParts());
#endif
    // Loop through all the nodes on the provided part number and compute the reconstruction.
    // First pull out the data we need
    const std::shared_ptr<FEMeshPartitioner>& mesh_partitioner = d_mesh_mapping->getMeshPartitioner(part);
    EquationSystems* eq_sys = mesh_partitioner->getEquationSystems();
    System& Q_sys = eq_sys->get_system(Q_str);
    NumericVector<double>* Q_vec = Q_sys.solution.get();
    const DofMap& Q_dof_map = Q_sys.get_dof_map();

    const MeshBase& mesh = eq_sys->get_mesh();
    auto it = mesh.local_nodes_begin();
    const auto it_end = mesh.local_nodes_end();
    for (; it != it_end; ++it)
    {
        const Node* const node = *it;
        std::vector<dof_id_type> Q_dof;
        Q_dof_map.dof_indices(node, Q_dof);
        Q_vec->set(Q_dof[0], reconstruct(part, node->id(), Q_idx));
    }

    Q_vec->close();
    Q_sys.update();
}

void
BoundaryReconstructCache::reconstruct(const std::string& Q_str, const int Q_idx)
{
    for (int part = 0; part < d_mesh_mapping->getNumParts(); ++part) reconstruct(part, Q_str, Q_idx);
}

const BoundaryReconstructCache::WeightStruct&
BoundaryReconstructCache::getWeightStruct(const int part, const int node_id) const
{
#ifndef NDEBUG
    TBOX_ASSERT(part < d_mesh_mapping->getNumParts());
    TBOX_ASSERT(node_id < static_cast<int>(d_mesh_mapping->getBoundaryMesh(part)->n_nodes()));
#endif
    return d_weights_map.at(std::make_pair(part, node_id));
}
} // namespace ADS
