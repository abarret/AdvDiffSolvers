CCAD_ADD_EXAMPLE(
  TARGET_NAME
    "fpc"
  OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/examples/fpc"
  OUTPUT_NAME
    main2d
  EXAMPLE_GROUP
    examples
  SOURCES
    main.cpp InsideLSFcn.cpp QFcn.cpp SurfaceBoundaryReactions.cpp
  LINK_TARGETS
    CCAD2d
  INPUT_FILES
    input2d
  )

