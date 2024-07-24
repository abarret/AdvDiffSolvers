## ---------------------------------------------------------------------
##
## Copyright (c) 2020 - 2020 by the IBAMR developers
## All rights reserved.
##
## This file is part of IBAMR.
##
## IBAMR is free software and is distributed under the 3-clause BSD
## license. The full text of the license can be found in the file
## COPYRIGHT at the top level directory of IBAMR.
##
## ---------------------------------------------------------------------

ADD_CUSTOM_TARGET(tests)

# We use the list of test directories in two ways:
# 1. each test is added to a target tests-dir so that, e.g., 'make tests-IBFE'
#    only compiles the IBFE tests.
# 2. all input and output files in these source directories are symlinked into
#    their corresponding build directories.
SET(TEST_DIRECTORIES adv_ops examples integrators operators level_set reconstructions sharp_interface surfaces utilities)

FOREACH(_dir ${TEST_DIRECTORIES})
  ADD_CUSTOM_TARGET("tests-${_dir}")
  ADD_DEPENDENCIES(tests "tests-${_dir}")
ENDFOREACH()

# TODO - we should clean up these macros so we don't have 3 macros that do
# basically the same thing

# Convenience macro that sets up an executable target which links against
# _target_link. For example, if the inputs are Foo, bar.cpp, and quux then we
# create a target Foo_bar in directory Foo which depends on quux.
MACRO(SETUP _dir _src _target_link)
  GET_FILENAME_COMPONENT(_out_name "${_src}" NAME_WE)
  SET(_target "tests-${_dir}_${_out_name}")
  ADD_EXECUTABLE(${_target} EXCLUDE_FROM_ALL "${_dir}/${_src}")
  SET_TARGET_PROPERTIES(${_target}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/tests/${_dir}"
    OUTPUT_NAME
    ${_out_name}
    )
  TARGET_COMPILE_OPTIONS(${_target} PUBLIC -DSOURCE_DIR="${CMAKE_SOURCE_DIR}/tests/${_dir}/")
  TARGET_LINK_LIBRARIES(${_target} PRIVATE "${_target_link}")
  ADD_DEPENDENCIES("tests-${_dir}" ${_target})
ENDMACRO()

# Like the last one, but for 2d targets. For example, if the inputs are Foo and
# bar.cpp then we create a target Foo_bar_2d in directory Foo which depends on
# IBAMR2d.
MACRO(SETUP_2D _dir _src)
  GET_FILENAME_COMPONENT(_dest "${_src}" NAME_WE)
  SET(_out_name "${_dest}_2d")
  SET(_target "tests-${_dir}_${_out_name}")
  ADD_EXECUTABLE(${_target} EXCLUDE_FROM_ALL "${_dir}/${_src}")
  SET_TARGET_PROPERTIES(${_target}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/tests/${_dir}"
    OUTPUT_NAME
    ${_out_name}
    )
  TARGET_COMPILE_OPTIONS(${_target} PUBLIC -DSOURCE_DIR="${CMAKE_SOURCE_DIR}/tests/${_dir}/")
  TARGET_LINK_LIBRARIES(${_target} PRIVATE ADS2d)
  ADD_DEPENDENCIES("tests-${_dir}" ${_target})
ENDMACRO()

# Like the last one, but for 3d targets. For example, if the inputs are Foo and
# bar.cpp then we create a target Foo_bar_3d in directory Foo which depends on
# IBAMR3d.
MACRO(SETUP_3D _dir _src)
  GET_FILENAME_COMPONENT(_dest "${_src}" NAME_WE)
  SET(_out_name "${_dest}_3d")
  SET(_target "tests-${_dir}_${_out_name}")
  ADD_EXECUTABLE(${_target} EXCLUDE_FROM_ALL "${_dir}/${_src}")
  SET_TARGET_PROPERTIES(${_target}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/tests/${_dir}"
    OUTPUT_NAME
    ${_out_name}
    )
  TARGET_COMPILE_OPTIONS(${_target} PUBLIC -DSOURCE_DIR="${CMAKE_SOURCE_DIR}/tests/${_dir}/")
  TARGET_LINK_LIBRARIES(${_target} PRIVATE ADS3d)
  ADD_DEPENDENCIES(tests-${_dir} ${_target})
ENDMACRO()

SETUP_2D(adv_ops divergence.cpp)

SETUP_2D(examples compressible.cpp)
SETUP_2D(examples disk.cpp)
SETUP_2D(examples peristalsis.cpp)
SETUP_2D(examples radial.cpp)
SETUP_2D(examples semi_lagrangian_01.cpp)

SETUP_2D(level_set extrapolated_integrator.cpp)
SETUP_2D(level_set internal_bdry_fill.cpp)
SETUP_2D(level_set internal_bdry_fill_2.cpp)
SETUP_2D(level_set find_vol.cpp)
SETUP_2D(level_set reinitialize_level_set.cpp)

SETUP_2D(operators boundary_reconstruction_cache.cpp)
SETUP_2D(operators laplace.cpp)
SETUP_2D(operators reconstructions.cpp)

SETUP_2D(reconstructions divergence.cpp)

SETUP_2D(sharp_interface classify_points_ls.cpp)
SETUP_2D(sharp_interface classify_points_struct.cpp)
SETUP_2D(sharp_interface find_image_points.cpp)

SETUP_2D(surfaces surface_jacobian.cpp)

SETUP_2D(utilities pointwise_function.cpp)

SETUP_3D(utilities pointwise_function.cpp)

SETUP_2D(integrators implicit_integrator.cpp)

ADD_CUSTOM_COMMAND(TARGET tests
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_SOURCE_DIR}/attest ${CMAKE_BINARY_DIR}/attest)

# Set up the input and output files. Since the input and output files aren't
# really used by the build system we use a shell script to find them every time
# 'make test' is run rather than evaluating the glob when cmake generates the
# build system.
FOREACH(_dir ${TEST_DIRECTORIES})
  ADD_CUSTOM_COMMAND(TARGET "tests-${_dir}"
    POST_BUILD
    COMMAND bash ${CMAKE_SOURCE_DIR}/tests/link-test-files.sh
    ${CMAKE_SOURCE_DIR}/tests/${_dir} ${CMAKE_BINARY_DIR}/tests/${_dir}
    VERBATIM)
ENDFOREACH()

# Find numdiff, if possible (we only need it for tests so its not essential that
# we find it now)
FIND_PROGRAM(NUMDIFF_EXECUTABLE NAMES numdiff HINTS ${NUMDIFF_ROOT} PATH_SUFFIXES bin)

IF ("${NUMDIFF_EXECUTABLE}" STREQUAL "NUMDIFF_EXECUTABLE-NOTFOUND")
  MESSAGE(WARNING "\
The configuration script was not able to locate numdiff. If you want to run \
the test suite you will need to either edit attest.conf, specify the path to \
numdiff to attest, or rerun CMake with the argument NUMDIFF_ROOT specifying \
numdiff's root installation directory.")
  # clear the value so that attest.conf doesn't contain an invalid path
  SET(NUMDIFF_EXECUTABLE "")
ENDIF()

# Set up the default attest configuration file:
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/tests/attest.conf.in
  ${CMAKE_BINARY_DIR}/attest.conf)
