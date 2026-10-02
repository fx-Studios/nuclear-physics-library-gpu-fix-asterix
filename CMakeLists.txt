cmake_minimum_required(VERSION 3.20)

# ─────────────────────────────────────────────────────────────────────
# Computation mode selection: CPU, GPU, or HYBRID.
#
# This MUST be parsed before project() so that we can set
# CMAKE_Fortran_COMPILER to nvfortran BEFORE CMake runs compiler
# detection. Setting it after project() has no effect — CMake has
# already locked in the default compiler (e.g. gfortran / f95).
#
#   CPU    — gfortran + OpenMP + OpenBLAS. Default, no GPU required.
#   GPU    — nvfortran + CUDA Fortran. All eligible work on GPU.
#   HYBRID — nvfortran + CUDA Fortran. Work split GPU/CPU by size.
#
# Usage:
#   cmake -DCOMPUTATION_MODE=CPU ..      # default
#   cmake -DCOMPUTATION_MODE=GPU ..
#   cmake -DCOMPUTATION_MODE=HYBRID ..
# ─────────────────────────────────────────────────────────────────────
set(COMPUTATION_MODE "CPU" CACHE STRING "Computation mode: CPU, GPU, or HYBRID")
set_property(CACHE COMPUTATION_MODE PROPERTY STRINGS CPU GPU HYBRID)

# Backward compatibility: BUILD_WITH_CUDA=ON maps to COMPUTATION_MODE=GPU.
if(DEFINED BUILD_WITH_CUDA AND BUILD_WITH_CUDA AND COMPUTATION_MODE STREQUAL "CPU")
    set(COMPUTATION_MODE "GPU")
    message(STATUS "BUILD_WITH_CUDA=ON is deprecated; using COMPUTATION_MODE=GPU")
endif()

# If GPU or HYBRID mode, find nvfortran and set CMAKE_Fortran_COMPILER
# BEFORE project() runs compiler detection. This is the only way to
# guarantee the nvfortran-specific flags (-cuda, -Mbackslash) reach the
# correct compiler.
if(COMPUTATION_MODE STREQUAL "GPU" OR COMPUTATION_MODE STREQUAL "HYBRID")
    find_program(NVFORTRAN_EXECUTABLE nvfortran)
    if(NOT NVFORTRAN_EXECUTABLE)
        message(FATAL_ERROR
            "COMPUTATION_MODE=${COMPUTATION_MODE} requires nvfortran. Install the NVIDIA "
            "HPC SDK (https://developer.nvidia.com/hpc-sdk) and ensure "
            "nvfortran is on PATH, or set -DCMAKE_Fortran_COMPILER=/path/to/nvfortran.")
    endif()

    # Force the Fortran compiler to nvfortran unless the user explicitly
    # overrode it on the command line.
    if(NOT CMAKE_Fortran_COMPILER)
        set(CMAKE_Fortran_COMPILER ${NVFORTRAN_EXECUTABLE} CACHE FILEPATH "Fortran compiler" FORCE)
    endif()
    message(STATUS "Computation mode: ${COMPUTATION_MODE} (nvfortran: ${NVFORTRAN_EXECUTABLE})")
else()
    message(STATUS "Computation mode: CPU-only (set -DCOMPUTATION_MODE=GPU or HYBRID to enable CUDA)")
endif()

# Now that CMAKE_Fortran_COMPILER is set correctly, enable languages.
project(NuclearPhysics VERSION 0.2.17 LANGUAGES Fortran C)

# Set Fortran standards and flags.
#
# CMAKE_Fortran_MODULE_DIRECTORY is declared as a CACHE variable so that
# every subdirectory (core/, kernels/, tests/, ...) sees the same value and
# a stale command-line -D override can't silently re-point it.
#
# CRITICAL for NVHPC (nvfortran): unlike gfortran, nvfortran does NOT
# auto-create the module output directory. If the directory does not
# exist when nvfortran runs, it silently writes NO .mod file and exits 0,
# which then triggers CMake's fatal:
#   Error copying Fortran module "mod/kinds.mod".
#     Tried "mod/KINDS.mod" and "mod/kinds.mod".
# We pre-create the directory at configure time below, AND we force the
# path to absolute -- nvfortran's -module flag with a relative path is
# another known source of "module not found" errors when make's CWD
# differs from the build directory.
get_filename_component(_NPL_BINARY_DIR_ABS "${PROJECT_BINARY_DIR}" ABSOLUTE)
set(CMAKE_Fortran_MODULE_DIRECTORY ${_NPL_BINARY_DIR_ABS}/mod CACHE PATH
    "Directory where Fortran .mod files are written and read (must be absolute for nvfortran)")
file(MAKE_DIRECTORY "${CMAKE_Fortran_MODULE_DIRECTORY}")
include_directories(${CMAKE_Fortran_MODULE_DIRECTORY})

# Safety net: re-create the mod/ directory before every build. file(MAKE_DIRECTORY)
# above only runs at configure time, so `make clean` or a fresh checkout could
# remove it and break the next nvfortran invocation. This custom target runs on
# every build and is idempotent.
add_custom_target(npl_ensure_mod_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_Fortran_MODULE_DIRECTORY}"
    COMMENT "Ensuring Fortran module directory exists: ${CMAKE_Fortran_MODULE_DIRECTORY}")

set(CMAKE_CONFIGURATION_TYPES "Debug;Release;Profiling" CACHE STRING "" FORCE)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -std=f2008 -fimplicit-none -Wall -Wextra -ffree-line-length-none -cpp")
    set(CMAKE_Fortran_FLAGS_DEBUG "-O0 -g -fcheck=all -fbacktrace -fimplicit-none -fno-omit-frame-pointer -pedantic ")
    set(CMAKE_Fortran_FLAGS_RELEASE "-Ofast -march=native -mtune=native -flto -ffpe-summary=none -fopenmp")
    set(CMAKE_Fortran_FLAGS_PROFILING "-O3 -g -pg -fopenmp")
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "PGI|NVHPC")
    # nvfortran-specific flags.
    #   -cuda     enables CUDA Fortran syntax (.cuf kernels, cudafor module)
    #   -Mbackslash  keeps backslash as a literal char (matches gfortran)
    #   -Mpreprocess  enables preprocessing so #ifdef works in .f90 files
    #   -Mstandard  enables Fortran standard-conformance checking
    # NOTE: nvfortran does NOT understand -std=f2008 (that's gfortran syntax).
    #       nvfortran accepts F2008 features by default; use -Mstandard to
    #       warn on non-standard extensions, or omit it for permissive mode.
    # -Mcpp is preprocess only and emits no object
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -Mbackslash -Mpreprocess")
    # cuda only where kernels are built
    if(COMPUTATION_MODE STREQUAL "GPU" OR COMPUTATION_MODE STREQUAL "HYBRID")
        set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -cuda")
    endif()
    set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -fast")
    set(CMAKE_Fortran_FLAGS_DEBUG "-O0 -g -C -traceback -Kcheck")
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -stand f08 -implicitnone -warn all -extend-source")
    set(CMAKE_Fortran_FLAGS_DEBUG "-g -check all -traceback")
    set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -xHost")
endif()

message(STATUS "Fortran compiler: ${CMAKE_Fortran_COMPILER} (${CMAKE_Fortran_COMPILER_ID})")

add_compile_definitions(_POSIX_C_SOURCE=200809L)
add_compile_definitions(M_PI=3.14159265358979)

# Option to manually specify OpenBLAS path
set(OPENBLAS_PATH "/usr/lib/libopenblas.so" CACHE FILEPATH "Path to OpenBLAS library")

# Try to find BLAS/LAPACK using CMake's built-in finders
# First, try to explicitly use OpenBLAS
set(BLA_VENDOR OpenBLAS)
find_package(BLAS QUIET)
find_package(LAPACK QUIET)

# If not found, try generic search
if(NOT BLAS_FOUND OR NOT LAPACK_FOUND)
    set(BLA_VENDOR All)
    find_package(BLAS QUIET)
    find_package(LAPACK QUIET)
endif()

# If still not found, use manual fallback to OpenBLAS
if(NOT BLAS_FOUND)
    if(EXISTS "${OPENBLAS_PATH}")
        set(BLAS_LIBRARIES "${OPENBLAS_PATH}")
        set(BLAS_FOUND TRUE)
        message(STATUS "Manually set BLAS to OpenBLAS: ${BLAS_LIBRARIES}")
    else()
        message(FATAL_ERROR "BLAS not found! Please install OpenBLAS or set OPENBLAS_PATH")
    endif()
endif()

if(NOT LAPACK_FOUND)
    if(EXISTS "${OPENBLAS_PATH}")
        set(LAPACK_LIBRARIES "${OPENBLAS_PATH}")
        set(LAPACK_FOUND TRUE)
        message(STATUS "Manually set LAPACK to OpenBLAS: ${LAPACK_LIBRARIES}")
    else()
        message(FATAL_ERROR "LAPACK not found! Please install OpenBLAS or set OPENBLAS_PATH")
    endif()
endif()

# Diagnostic output
message(STATUS "BLAS libraries: ${BLAS_LIBRARIES}")
message(STATUS "LAPACK libraries: ${LAPACK_LIBRARIES}")

# Find OpenGL and GLUT (optional for visualisation)
option(BUILD_WITH_OPENGL "Build OpenGL visualisation examples" ON)

if(BUILD_WITH_OPENGL)
    find_package(OpenGL)
    find_package(GLUT)
    
    if(OpenGL_FOUND AND GLUT_FOUND)
        message(STATUS "OpenGL found: ${OPENGL_LIBRARIES}")
        message(STATUS "GLUT found: ${GLUT_LIBRARIES}")
        set(OPENGL_AVAILABLE TRUE)
    else()
        message(STATUS "OpenGL or GLUT not found - OpenGL examples will not be built")
        message(STATUS "  To enable, install: sudo apt-get install freeglut3-dev mesa-common-dev")
        set(OPENGL_AVAILABLE FALSE)
    endif()
else()
    set(OPENGL_AVAILABLE FALSE)
    message(STATUS "OpenGL support disabled by user")
endif()

# ─────────────────────────────────────────────────────────────────────
# Post-project CUDA configuration (part 1: extension registration).
#
# The compiler selection (nvfortran vs gfortran) and COMPUTATION_MODE
# parsing happen BEFORE project() above. Here we:
#   1. Register .cuf as a Fortran source extension (for module scanning)
#   2. Set the preprocessor macro for the selected mode
#
# The actual add_subdirectory(cuda) is deferred to AFTER add_subdirectory(core)
# below, because the .cuf files `use kinds` and CMake's module scanner needs
# the core target (which produces kinds.mod) to be defined first.
# ─────────────────────────────────────────────────────────────────────
if(COMPUTATION_MODE STREQUAL "GPU" OR COMPUTATION_MODE STREQUAL "HYBRID")
    # Register .cuf as a native Fortran source extension so CMake's
    # dependency scanner properly tracks module/use statements across
    # .cuf files. Without this, the scanner skips .cuf files and
    # produces a malformed depend.make (module paths get concatenated),
    # causing "Error copying Fortran module" at build time.
    # This MUST be set after project() (which initializes the default
    # extension list) but before any add_subdirectory that contains .cuf.
    if(NOT "cuf" IN_LIST CMAKE_Fortran_SOURCE_FILE_EXTENSIONS)
        list(APPEND CMAKE_Fortran_SOURCE_FILE_EXTENSIONS "cuf")
    endif()
    message(STATUS "Fortran source extensions: ${CMAKE_Fortran_SOURCE_FILE_EXTENSIONS}")

    if(COMPUTATION_MODE STREQUAL "GPU")
        add_compile_definitions(COMPUTATION_MODE_GPU)
    else()
        add_compile_definitions(COMPUTATION_MODE_HYBRID)
    endif()
else()
    add_compile_definitions(COMPUTATION_MODE_CPU)
endif()

# Core library — must be defined BEFORE cuda/ because the .cuf files
# `use kinds` and CMake's Fortran module scanner needs to know where
# kinds.mod will be generated.
add_subdirectory(core)

# Make every Fortran target in the project depend on the mod/ directory
# existing. This is critical for nvfortran (NVHPC), which does NOT auto-create
# the module output directory and silently produces no .mod file if it is
# missing -- causing CMake's post-compile cmake_copy_f90_mod step to fail
# with "Error copying Fortran module".
add_dependencies(nuclear_physics_core npl_ensure_mod_dir)

# Kernel library (depends on core)
add_subdirectory(kernels)
add_dependencies(nuclear_physics_kernels npl_ensure_mod_dir)
# Link core into kernels. LAPACK/BLAS are linked inside kernels/CMakeLists.txt
# (the kernels library calls dgesv directly in backward_euler.f90 and eigen.f90).
target_link_libraries(nuclear_physics_kernels PUBLIC nuclear_physics_core)

# ─────────────────────────────────────────────────────────────────────
# CUDA library (part 2: add_subdirectory).
# Deferred to here (after add_subdirectory(core)) because the .cuf files
# `use kinds` and CMake's Fortran module scanner needs the core target
# (which produces kinds.mod) to be already defined so it can record
# the inter-target module dependency.
# ─────────────────────────────────────────────────────────────────────
if(COMPUTATION_MODE STREQUAL "GPU" OR COMPUTATION_MODE STREQUAL "HYBRID")
    add_subdirectory(cuda)
endif()

add_subdirectory(tests)
add_subdirectory(models)
target_link_libraries(nuclear_physics_models PUBLIC nuclear_physics_kernels nuclear_physics_core)

# When CUDA is enabled, link the GPU acceleration library into models so
# that the model source files can call GPU kernels directly via #ifdef.
if(COMPUTATION_MODE STREQUAL "GPU" OR COMPUTATION_MODE STREQUAL "HYBRID")
    target_link_libraries(nuclear_physics_models PUBLIC nuclear_physics_cuda)
    target_link_options(nuclear_physics_models PRIVATE -cuda)
endif()

# Subsystems library — one module per spec chapter (vessel, fuel, crd,
# recirculation, main_steam, feedwater) implementing the 5-call contract
# from REORG_PLAN §3. Step 2 wires only the skeleton; subsystem-level
# behaviour migrates in steps 3-7.
add_subdirectory(subsystems)
# Step 11 (reorg): fuel.f90 imports two_phase_flow for the heat↔two-phase
# convection-coefficient helper, so subsystems must compile after models.
target_link_libraries(nuclear_physics_subsystems nuclear_physics_models nuclear_physics_core)

# C Interface library
add_subdirectory(interface)

# BWR Core Simulator with OpenGL (if available)
if(OPENGL_AVAILABLE)
    add_executable(bwr_core_simulator_opengl examples/bwr_core_simulator_opengl.f90)
    
    target_link_libraries(bwr_core_simulator_opengl 
        nuclear_physics_models 
        nuclear_physics_kernels 
        nuclear_physics_core
        ${LAPACK_LIBRARIES} 
        ${BLAS_LIBRARIES}
        ${OPENGL_LIBRARIES}
        ${GLUT_LIBRARIES})

    # When CUDA is enabled (GPU or HYBRID mode), link the GPU acceleration
    # library and pull in the cudafor module + libcudaclient runtime
    # (nvfortran links this automatically when it sees -cuda on the link line).
    if(COMPUTATION_MODE STREQUAL "GPU" OR COMPUTATION_MODE STREQUAL "HYBRID")
        target_link_libraries(bwr_core_simulator_opengl nuclear_physics_cuda)
        target_compile_definitions(bwr_core_simulator_opengl PRIVATE USE_CUDA)
        target_link_options(bwr_core_simulator_opengl PRIVATE -cuda)
    endif()
    
    target_include_directories(bwr_core_simulator_opengl PRIVATE 
        include
        ${OPENGL_INCLUDE_DIR}
        ${GLUT_INCLUDE_DIR})
    
    # Add definitions for OpenGL
    target_compile_definitions(bwr_core_simulator_opengl PRIVATE USE_OPENGL)
    
    install(TARGETS bwr_core_simulator_opengl RUNTIME DESTINATION bin)
    
    message(STATUS "OpenGL visualisation enabled - bwr_core_simulator_opengl will be built")
endif()

add_executable(bwr_core_simulator examples/bwr_core_simulator.f90)

target_link_libraries(bwr_core_simulator
    nuclear_physics_models
    nuclear_physics_kernels
    nuclear_physics_core
    ${LAPACK_LIBRARIES}
    ${BLAS_LIBRARIES})

if(COMPUTATION_MODE STREQUAL "GPU" OR COMPUTATION_MODE STREQUAL "HYBRID")
    target_link_libraries(bwr_core_simulator nuclear_physics_cuda)
    target_compile_definitions(bwr_core_simulator PRIVATE USE_CUDA)
    target_link_options(bwr_core_simulator PRIVATE -cuda)
endif()

target_include_directories(bwr_core_simulator PRIVATE include)

install(TARGETS bwr_core_simulator RUNTIME DESTINATION bin)

# Export configuration
install(EXPORT NuclearPhysicsTargets
    FILE NuclearPhysicsTargets.cmake
    NAMESPACE NuclearPhysics::
    DESTINATION lib/cmake/NuclearPhysics)

# Create package config
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/NuclearPhysicsConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/NuclearPhysicsConfig.cmake
    INSTALL_DESTINATION lib/cmake/NuclearPhysics)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/NuclearPhysicsConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/NuclearPhysicsConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/NuclearPhysicsConfigVersion.cmake
    DESTINATION lib/cmake/NuclearPhysics)

# Installation summary
message(STATUS "")
message(STATUS "")
message(STATUS "  Nuclear Physics Library Configuration")
message(STATUS "═════════════════════════════════════════════════════")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Computation mode: ${COMPUTATION_MODE}")
message(STATUS "  Fortran compiler: ${CMAKE_Fortran_COMPILER_ID}")
message(STATUS "  C compiler: ${CMAKE_C_COMPILER_ID}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
message(STATUS "  Libraries:")
message(STATUS "    - nuclear_physics_core")
message(STATUS "    - nuclear_physics_kernels")
message(STATUS "    - nuclear_physics_models")
message(STATUS "    - nuclear_physics_subsystems")
message(STATUS "    - nuclear_physics_c (C interface)")
if(COMPUTATION_MODE STREQUAL "GPU" OR COMPUTATION_MODE STREQUAL "HYBRID")
    message(STATUS "    - nuclear_physics_cuda (CUDA Fortran acceleration)")
endif()
message(STATUS "")
message(STATUS "  Examples:")
if(OPENGL_AVAILABLE)
    message(STATUS "    - bwr_core_simulator_opengl (3D visualisation)")
else()
    message(STATUS "    - bwr_core_simulator_opengl (not available - missing OpenGL/GLUT)")
endif()
if(COMPUTATION_MODE STREQUAL "GPU" OR COMPUTATION_MODE STREQUAL "HYBRID")
    message(STATUS "    - bwr_core_simulator_opengl links nuclear_physics_cuda (GPU acceleration)")
endif()
message(STATUS "═════════════════════════════════════════════════════")
message(STATUS "")