ADS_ADD_EXAMPLE(
  TARGET_NAME
    "level_set-ex2"
  OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/examples/level_set/ex2"
  OUTPUT_NAME
    main2d
  EXAMPLE_GROUP
    examples
  SOURCES
  main.cpp IBBoundaryMeshMapping.cpp QFcn.cpp LSCUIConvectiveOperator.cpp
  LINK_TARGETS
    ADS2d
  INPUT_FILES
    input2d
  )

