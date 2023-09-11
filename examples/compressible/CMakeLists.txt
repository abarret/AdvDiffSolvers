ADS_ADD_EXAMPLE(
  TARGET_NAME
    "compressible"
  OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/examples/compressible"
  OUTPUT_NAME
    main2d
  EXAMPLE_GROUP
    examples
  SOURCES
    main.cpp QFcn.cpp
  LINK_TARGETS
    ADS2d
  INPUT_FILES
    input2d
  )

