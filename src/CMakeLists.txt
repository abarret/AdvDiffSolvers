ADD_LIBRARY(ADS2d)
ADD_LIBRARY(ADS3d)

SET(FORTRAN_SRC_BASE
	fortran/advection_routines)

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
    AdvectiveReconstructionOperator.cpp
    CutCellMeshMapping.cpp
    CutCellVolumeMeshMapping.cpp
    FDWeightsCache.cpp
    FEMeshPartitioner.cpp
    GeneralBoundaryMeshMapping.cpp
    GhostPoints.cpp
    GlobalIndexing.cpp
    IntegrateFunction.cpp
    LinearReconstructions.cpp
    LSAdvDiffIntegrator.cpp
    LSCartGridFunction.cpp
    LSFindCellVolume.cpp
    LSFromLevelSet.cpp
    LSFromMesh.cpp
    ls_functions.cpp
    MLSReconstructCache.cpp
    RBFFDWeightsCache.cpp
    RBFReconstructCache.cpp
    RBFReconstructions.cpp
    ReconstructCache.cpp
    reconstructions.cpp
    SBAdvDiffIntegrator.cpp
    SBIntegrator.cpp
    SBSurfaceFluidCouplingManager.cpp
    VolumeBoundaryMeshMapping.cpp
    ZSplineReconstructions.cpp

    solvers/ConditionCounter.cpp
    solvers/LSCutCellBoundaryConditions.cpp
    solvers/LSCutCellLaplaceOperator.cpp
    solvers/PETScAugmentedKrylovLinearSolver.cpp
    solvers/PETScAugmentedLinearOperator.cpp
    solvers/RBFFDPoissonSolver.cpp
    solvers/RBFLaplaceOperator.cpp
    solvers/SBBoundaryConditions.cpp

    rbffd/PointList.cpp

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
