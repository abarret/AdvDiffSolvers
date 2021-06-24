CCAD_ADD_EXAMPLE(
  TARGET_NAME
    "reacting_couette"
  OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/examples/reacting_couette"
  OUTPUT_NAME
    main2d
  EXAMPLE_GROUP
    examples
  SOURCES
    main.cpp InsideBoundaryConditions.cpp InsideLSFcn.cpp OutsideBoundaryConditions.cpp OutsideLSFcn.cpp QFcn.cpp
  LINK_TARGETS
    CCAD2d
  INPUT_FILES
    input2d
  )

CCAD_ADD_EXAMPLE(
  TARGET_NAME
    "reacting_couette_convergence"
  OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/examples/reacting_couette"
  OUTPUT_NAME
    convergence_tester2d
  EXAMPLE_GROUP
    examples
  SOURCES
    convergence_tester.cpp InsideBoundaryConditions.cpp InsideLSFcn.cpp OutsideBoundaryConditions.cpp OutsideLSFcn.cpp QFcn.cpp
  LINK_TARGETS
    CCAD2d
  INPUT_FILES
    input2d.convergence
  )
