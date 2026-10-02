# CMakeLists.txt for test suite
#
# This subdirectory builds a standalone test executable with aggressive
# debug flags (sanitizers, bounds checking). These flags are gfortran-
# specific, so the test suite is only built in CPU mode. In GPU or
# HYBRID mode (which uses nvfortran), the test build is skipped because:
#   1. nvfortran does not support -fsanitize=address
#   2. CMake's FindBLAS may not detect BLAS for nvfortran
#   3. The sanitizer flags would conflict with -cuda
#
# To run tests in GPU/HYBRID mode, build the main library with nvfortran
# and write separate tests that don't require sanitizer instrumentation.

# Skip the test build entirely in GPU/HYBRID mode.
if(DEFINED COMPUTATION_MODE AND NOT COMPUTATION_MODE STREQUAL "CPU")
    message(STATUS "Test suite skipped in COMPUTATION_MODE=${COMPUTATION_MODE}")
    return()
endif()
# these flags are gfortran only
if(NOT CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
    message(STATUS "Test suite skipped: needs gfortran for the sanitizer/bounds-check "
                   "flags below (compiler is ${CMAKE_Fortran_COMPILER_ID})")
    return()
endif()

# NOTE: do NOT call project() here. Calling project() in a subdirectory
# re-runs Fortran compiler detection, resets several CMAKE_Fortran_*
# cache entries, and -- critically -- creates a new PROJECT_BINARY_DIR
# scope. Combined with overriding CMAKE_Fortran_MODULE_DIRECTORY below
# this previously caused:
#   Error copying Fortran module "mod/kinds.mod".
#     Tried "mod/KINDS.mod" and "mod/kinds.mod".
# because two targets (nuclear_physics_core and the test 'core') both
# claimed to provide module 'kinds' from overlapping-but-different
# directories, and CMake's module-copy step looked for the .mod file in
# a directory the compiler had not written to.
#
# We reuse the parent project's compiler detection and module directory
# verbatim, and only override the *flags* (which is safe).

# Set Fortran standard. These flags are gfortran-specific (sanitizers,
# bounds checking) and only valid in CPU mode.
# -cpp needed for the #if guards in test_suite.f90
set(CMAKE_Fortran_FLAGS "-cpp -g3 -O0 -fcheck=all -fbacktrace -Wall -Wextra -fsanitize=address -fno-omit-frame-pointer -fno-inline -fno-lto -fsanitize=address,undefined -fstack-protector-strong")

# Reuse the parent project's module directory for *reading* .mod files
# produced by the main library, but write test-built .mod files into a
# separate directory. This is critical: the test 'core' target compiles
# the same kinds.f90 as nuclear_physics_core, but with sanitizer flags.
# If both write to the same ${PROJECT_BINARY_DIR}/mod/kinds.mod, the two
# builds race and one of them sees a half-written or stale .mod file,
# which triggers CMake's "Error copying Fortran module" fatal message.
set(TESTS_MODULE_DIR ${CMAKE_BINARY_DIR}/test_modules CACHE PATH
    "Module directory for the test-suite targets (kept separate from the main library)")
file(MAKE_DIRECTORY "${TESTS_MODULE_DIR}")
include_directories(BEFORE ${TESTS_MODULE_DIR} ${CMAKE_Fortran_MODULE_DIRECTORY})

# Find BLAS and LAPACK. Reuse the top-level findings if available;
# otherwise do our own search.
if(DEFINED BLAS_LIBRARIES AND DEFINED LAPACK_LIBRARIES)
    set(BLAS_FOUND TRUE)
    set(LAPACK_FOUND TRUE)
else()
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
endif()

# Core library
add_library(core
    ../core/kinds.F90
    ../core/constants.f90
    ../core/numeric_utils.f90
    ../core/rng.f90
    ../core/compute_mode.F90
)
set_target_properties(core PROPERTIES Fortran_MODULE_DIRECTORY ${TESTS_MODULE_DIR})

# Linear algebra library (depends on BLAS/LAPACK)
add_library(linalg
    ../kernels/linear_algebra/dense_matrix.f90
    ../kernels/linear_algebra/eigen.f90
    ../kernels/linear_algebra/solve_linear.f90
    ../kernels/linear_algebra/sparse_matrix.f90
)
set_target_properties(linalg PROPERTIES Fortran_MODULE_DIRECTORY ${TESTS_MODULE_DIR})
target_link_libraries(linalg core ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

# ODE solvers
add_library(ode
    ../kernels/ode/rk4.f90
    ../kernels/ode/dormand_prince.f90
    ../kernels/ode/backward_euler.f90
)
set_target_properties(ode PROPERTIES Fortran_MODULE_DIRECTORY ${TESTS_MODULE_DIR})
target_link_libraries(ode core linalg)

# Optimisation
add_library(optimisation
    ../kernels/optimisation/gradient_descent.f90
    ../kernels/optimisation/conjugate_newton.f90
    ../kernels/optimisation/quasi_newton.f90
    ../kernels/optimisation/constrained.f90
)
set_target_properties(optimisation PROPERTIES Fortran_MODULE_DIRECTORY ${TESTS_MODULE_DIR})
target_link_libraries(optimisation core)

# PDE solvers
add_library(pde
    ../kernels/pde/finite_difference.f90
    ../kernels/pde/finite_volume.f90
    ../kernels/pde/spectral.f90
)
set_target_properties(pde PROPERTIES Fortran_MODULE_DIRECTORY ${TESTS_MODULE_DIR})
target_link_libraries(pde core)

# Physics models (lives under physics/ after the reorg step-1 rename).
add_library(models
    ../models/heat_transfer.F90
    ../models/two_phase_flow.F90
    ../models/multigroup_diffusion.F90
    ../models/cross_sections.F90
    ../models/burnup_depletion.F90
)
set_target_properties(models PROPERTIES Fortran_MODULE_DIRECTORY ${TESTS_MODULE_DIR})
target_link_libraries(models core linalg pde)

# Subsystems library (reorg step 2). Adds modules incrementally as each
# reorg step lights them up; the other modules still compile via the
# top-level build.
add_library(subsystems_test
    ../subsystems/vessel.f90
    ../subsystems/crd.f90
    ../subsystems/main_steam.f90
    ../subsystems/feedwater.f90
    ../subsystems/recirculation.f90
    ../subsystems/fuel.f90
    ../subsystems/rhr.f90
    ../subsystems/condensate_loop.f90
)
set_target_properties(subsystems_test PROPERTIES Fortran_MODULE_DIRECTORY ${TESTS_MODULE_DIR})
target_link_libraries(subsystems_test core linalg models pde)

# Test executable
add_executable(test_suite test_suite.f90)
set_target_properties(test_suite PROPERTIES Fortran_MODULE_DIRECTORY ${TESTS_MODULE_DIR})
target_link_options(test_suite PRIVATE "-Wl,-z,noexecstack")
target_link_libraries(test_suite
    core
    linalg
    ode
    optimisation
    pde
    models
    subsystems_test
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)

# Enable testing
enable_testing()
add_test(NAME FullTestSuite COMMAND test_suite)

# Custom target to run tests
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --verbose
    DEPENDS test_suite
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Print configuration
message(STATUS "Fortran Compiler: ${CMAKE_Fortran_COMPILER}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Test Module Directory: ${TESTS_MODULE_DIR}")
message(STATUS "Main Library Module Directory: ${CMAKE_Fortran_MODULE_DIRECTORY}")
