ADD_CUSTOM_TARGET(examples)
SET(EXAMPLE_DIRECTORIES couette disk pipe_flow_no_ins radial reacting_couette rotational)

MACRO(ADS_ADD_EXAMPLE)
  SET(OPTIONS)

  SET(ONE_VALUE_ARGS TARGET_NAME OUTPUT_DIRECTORY OUTPUT_NAME EXAMPLE_GROUP)
  SET(MULTI_VALUE_ARGS SOURCES REQUIRES LINK_TARGETS INPUT_FILES EXTRA_FILES)
  CMAKE_PARSE_ARGUMENTS(EX "${OPTIONS}" "${ONE_VALUE_ARGS}" "${MULTI_VALUE_ARGS}" ${ARGN})

  SET(_failed_requirement)
  SET(_requirements_met TRUE)
  FOREACH(_requirement ${EX_REQUIRES})
    IF(NOT ${${_requirement}})
      SET(_requirements_met FALSE)
      SET(_failed_requirement ${_requirement})
    ENDIF()
  ENDFOREACH()

  IF(${_requirements_met})
    ADD_EXECUTABLE("${EX_TARGET_NAME}" EXCLUDE_FROM_ALL ${EX_SOURCES})
    SET_TARGET_PROPERTIES("${EX_TARGET_NAME}"
      PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY
      "${EX_OUTPUT_DIRECTORY}"
      OUTPUT_NAME
      "${EX_OUTPUT_NAME}")
    TARGET_INCLUDE_DIRECTORIES("${EX_TARGET_NAME}" PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

    IF("${EX_LINK_TARGETS}" STREQUAL "")
      # If we have no target link libraries we need to set up the C++ standard ourselves
      TARGET_COMPILE_FEATURES("${EX_TARGET_NAME}" PUBLIC cxx_std_11)
    ELSE()
      TARGET_LINK_LIBRARIES("${EX_TARGET_NAME}" PRIVATE ${EX_LINK_TARGETS})
    ENDIF()
    ADD_DEPENDENCIES("${EX_EXAMPLE_GROUP}" "${EX_TARGET_NAME}")

    FOREACH(_input_file ${EX_INPUT_FILES})
      CONFIGURE_FILE("${_input_file}" "${EX_OUTPUT_DIRECTORY}" COPYONLY)
    ENDFOREACH()

    FOREACH(_extra_file ${EX_EXTRA_FILES})
      CONFIGURE_FILE("${_extra_file}" "${EX_OUTPUT_DIRECTORY}" COPYONLY)
    ENDFOREACH()
  ELSE()
    MESSAGE(WARNING "Example ${EX_TARGET_NAME} could not be set up since the "
      "required feature ${_failed_requirement} = FALSE")
  ENDIF()
ENDMACRO()

FOREACH(_dir ${EXAMPLE_DIRECTORIES})
  ADD_CUSTOM_TARGET("examples-${_dir}")
  ADD_DEPENDENCIES(examples "examples-${_dir}")
ENDFOREACH()

FOREACH(_dir ${EXAMPLE_DIRECTORIES})
  ADD_SUBDIRECTORY(${_dir})
ENDFOREACH()

