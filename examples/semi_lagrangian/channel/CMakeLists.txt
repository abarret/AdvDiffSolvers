ADS_ADD_EXAMPLE(
  TARGET_NAME
    "sl-channel"
  OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/examples/semi_lagrangian/channel"
  OUTPUT_NAME
    main2d
  EXAMPLE_GROUP
    examples-semi_lagrangian
  SOURCES
    main.cpp LSFcn.cpp QFcn.cpp
  LINK_TARGETS
    ADS2d
  INPUT_FILES
    input2d
  )

