ADS_ADD_EXAMPLE(
  TARGET_NAME
    "rotational"
  OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/examples/rotational"
  OUTPUT_NAME
    main2d
  EXAMPLE_GROUP
    examples
  SOURCES
    main.cpp LSFcn.cpp QFcn.cpp RadialBoundaryCond.cpp
  LINK_TARGETS
    ADS2d
  INPUT_FILES
    input2d
  )

ADS_ADD_EXAMPLE(
  TARGET_NAME
    "rotational_converge"
  OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/examples/rotational"
  OUTPUT_NAME
    convergence_tester2d
  EXAMPLE_GROUP
    examples
  SOURCES
    convergence_tester.cpp LSFcn.cpp QFcn.cpp RadialBoundaryCond.cpp
  LINK_TARGETS
    ADS2d
  INPUT_FILES
    input2d
  )

