CCAD_ADD_EXAMPLE(
  TARGET_NAME
    "disk"
  OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/examples/disk"
  OUTPUT_NAME
    main2d
  EXAMPLE_GROUP
    examples
  SOURCES
    main.cpp ForcingFcn.cpp QFcn.cpp
  LINK_TARGETS
    CCAD2d
  INPUT_FILES
    input2d
  )

