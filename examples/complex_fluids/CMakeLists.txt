ADS_ADD_EXAMPLE(
  TARGET_NAME
    "complex_fluids"
  OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/examples/complex_fluids"
  OUTPUT_NAME
    main2d
  EXAMPLE_GROUP
    examples
  SOURCES
    main.cpp VECFStrategy.cpp IBBoundaryMeshMapping.cpp
  LINK_TARGETS
    ADS2d
  INPUT_FILES
    input2d
  )

