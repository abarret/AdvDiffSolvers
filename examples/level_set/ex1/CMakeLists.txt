ADS_ADD_EXAMPLE(
  TARGET_NAME
    "level_set-ex1"
  OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/examples/level_set/ex1"
  OUTPUT_NAME
    main2d
  EXAMPLE_GROUP
    examples
  SOURCES
  main.cpp
  LINK_TARGETS
    ADS2d
  INPUT_FILES
    input2d
  )

