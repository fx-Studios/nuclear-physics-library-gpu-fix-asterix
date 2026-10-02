# interface/CMakeLists.txt
#
# Builds the C/C++ interface layer for the nuclear physics library
#

# Create the C interface shared library
add_library(nuclear_physics_c SHARED
    c_interface.f90
)

# Set Fortran module output directory to match main project
set_target_properties(nuclear_physics_c PROPERTIES
    Fortran_MODULE_DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY}
)

# Link against all required libraries.
# c_interface.f90 uses modules from nuclear_physics_subsystems (crd, vessel,
# main_steam, feedwater, recirculation, fuel, rhr, condensate_loop), so it
# MUST link the subsystems target — otherwise the build fails with
# "Cannot open module file 'crd.mod'" because subsystems haven't been
# compiled yet when c_interface.f90 is parsed.
target_link_libraries(nuclear_physics_c
    PUBLIC
        nuclear_physics_subsystems
        nuclear_physics_models
        nuclear_physics_kernels
        nuclear_physics_core
        ${LAPACK_LIBRARIES}
        ${BLAS_LIBRARIES}
)

# Install the C interface library
install(TARGETS nuclear_physics_c
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Install the C header file
install(FILES 
    ${PROJECT_SOURCE_DIR}/include/nuclear_physics.h
    DESTINATION include
)