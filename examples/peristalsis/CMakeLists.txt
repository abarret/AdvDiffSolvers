ADS_ADD_EXAMPLE(
  TARGET_NAME
    "peristalsis"
  OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/examples/peristalsis"
  OUTPUT_NAME
    main2d
  EXAMPLE_GROUP
    examples
  SOURCES
  main.cpp IBBoundaryMeshMapping.cpp QFcn.cpp
  LINK_TARGETS
    ADS2d
  INPUT_FILES
    input2d
  )

