CCAD_ADD_EXAMPLE(
  TARGET_NAME
    "pipe_flow_no_ins"
  OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/examples/pipe_flow_no_ins"
  OUTPUT_NAME
    main2d
  EXAMPLE_GROUP
    examples
  SOURCES
    main.cpp InsideLSFcn.cpp LSPipeFlow.cpp QFcn.cpp
  LINK_TARGETS
    CCAD2d
  INPUT_FILES
    input2d.pressure input2d.velocity
  )

CCAD_ADD_EXAMPLE(
  TARGET_NAME
    "pipe_flow_no_ins_converge"
  OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/examples/pipe_flow_no_ins"
  OUTPUT_NAME
    convergence_tester2d
  EXAMPLE_GROUP
    examples
  SOURCES
    convergence_tester.cpp InsideLSFcn.cpp LSPipeFlow.cpp QFcn.cpp
  LINK_TARGETS
    CCAD2d
  INPUT_FILES
    input2d.convergence
  )
