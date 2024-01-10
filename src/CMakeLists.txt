ADD_LIBRARY(ADS2d)
ADD_LIBRARY(ADS3d)

SET(FORTRAN_SRC_BASE
	fortran/advection_routines
	level_set/fortran/level_set_utils)

FOREACH(_d ${ADS_DIMENSIONS})
  FOREACH(_fortran_root_name ${FORTRAN_SRC_BASE})
    ADS_PROCESS_M4(NAME ${_fortran_root_name}${_d}d.f)
  ENDFOREACH()

  SET("FORTRAN_GENERATED_SRC${_d}D" "")
  FOREACH(_fortran_root_name ${FORTRAN_SRC_BASE})
    LIST(APPEND "FORTRAN_GENERATED_SRC${_d}D" "${CMAKE_CURRENT_BINARY_DIR}/${_fortran_root_name}${_d}d.f")
  ENDFOREACH()
ENDFOREACH()


SET(CXX_SRC
    
    adv_ops/AdvectiveReconstructionOperator.cpp
    adv_ops/InterpDivergenceReconstructions.cpp
    adv_ops/LagrangeReconstructions.cpp
    adv_ops/LagrangeStructureReconstructions.cpp
    adv_ops/LinearReconstructions.cpp
    adv_ops/RBFDivergenceReconstructions.cpp
    adv_ops/RBFReconstructions.cpp
    adv_ops/RBFStructureReconstructions.cpp
    adv_ops/ZSplineReconstructions.cpp
    
    cut_cell/CutCellMeshMapping.cpp
    cut_cell/CutCellVolumeMeshMapping.cpp

    integrators/ExtrapolatedAdvDiffHierarchyIntegrator.cpp
    integrators/LSAdvDiffIntegrator.cpp
    integrators/SBAdvDiffIntegrator.cpp
    integrators/SBIntegrator.cpp
    integrators/SLAdvIntegrator.cpp

    level_set/InternalBdryFill.cpp    
    level_set/LSCartGridFunction.cpp
    level_set/LSFindCellVolume.cpp
    level_set/LSFromLevelSet.cpp
    level_set/LSFromMesh.cpp
    level_set/ReinitializeLevelSet.cpp
    level_set/ls_functions.cpp
    
    rbffd/FDWeightsCache.cpp
    rbffd/GhostPoints.cpp
    rbffd/GlobalIndexing.cpp
    rbffd/PointList.cpp
    rbffd/RBFFDWeightsCache.cpp
    
    reconstructions/BoundaryReconstructCache.cpp
    reconstructions/MLSReconstructCache.cpp
    reconstructions/RBFReconstructCache.cpp
    reconstructions/ReconstructCache.cpp
    reconstructions/reconstructions.cpp

    solvers/ConditionCounter.cpp
    solvers/LSCutCellBoundaryConditions.cpp
    solvers/LSCutCellLaplaceOperator.cpp
    solvers/PETScAugmentedKrylovLinearSolver.cpp
    solvers/PETScAugmentedLinearOperator.cpp
    solvers/RBFFDPoissonSolver.cpp
    solvers/RBFLaplaceOperator.cpp
    solvers/SBBoundaryConditions.cpp
    
    surfaces/FEMeshPartitioner.cpp
    surfaces/GeneralBoundaryMeshMapping.cpp
    surfaces/SBSurfaceFluidCouplingManager.cpp
    surfaces/VolumeBoundaryMeshMapping.cpp
    surfaces/surface_utilities.cpp

    utilities/IntegrateFunction.cpp
    utilities/PointwiseFunction.cpp
    utilities/solver_utilities.cpp
    )

TARGET_SOURCES(ADS2d PRIVATE ${FORTRAN_GENERATED_SRC2D} ${CXX_SRC})
TARGET_SOURCES(ADS3d PRIVATE ${FORTRAN_GENERATED_SRC3D} ${CXX_SRC})

TARGET_INCLUDE_DIRECTORIES(ADS2d PUBLIC
	$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include/>
	$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/>)
TARGET_INCLUDE_DIRECTORIES(ADS3d PUBLIC
	$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include/>
	$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/>)

TARGET_LINK_LIBRARIES(ADS2d PUBLIC ADSHeaders)
TARGET_LINK_LIBRARIES(ADS3d PUBLIC ADSHeaders)

ADSLIB_SETUP_TARGET_LIBRARY(ADS2d)
ADSLIB_SETUP_TARGET_LIBRARY(ADS3d)

INSTALL(TARGETS ADS2d EXPORT ADSTargets COMPONENT library)
INSTALL(TARGETS ADS3d EXPORT ADSTargets COMPONENT library)
