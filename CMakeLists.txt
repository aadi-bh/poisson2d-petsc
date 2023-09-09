cmake_minimum_required(VERSION 3.26)
PROJECT(poisson2d CXX)
include(CTest)
include(CMakePrintHelpers)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 11)
set(MAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_VERBOSE_MAKEFILE TRUE)
set(CMAKE_VERBOSE_MAKEFILE FALSE)

# PkgConfig 
find_package(PkgConfig)

# MPI and PETSc
find_package(MPI REQUIRED)
if (PKG_CONFIG_FOUND)
    pkg_check_modules(PETSC PETSc)
endif()

if (PETSC_FOUND)
    list(APPEND COMPILE_OPTIONS ${PETSC_CFLAGS})

    include_directories(${PETSC_INCLUDE_DIRS})
    
    set(LINK_FLAGS "${LINK_FLAGS} ${PETSC_LDFLAGS}")   
    
    list(APPEND LIBRARIES ${PETSC_LINK_LIBRARIES})   
    
    set(CMAKE_REQUIRED_FLAGS ${PETSC_CFLAGS})   
    
    set(CMAKE_REQUIRED_INCLUDES "${PETSC_INCLUDE_DIRS}")
endif()
if (MPI_FOUND)
		list(APPEND COMPILE_OPTIONS ${MPI_CFLAGS})

		include_directories(${MPI_INCLUDE_PATH})

		set(LINK_FLAGS "${LINK_FLAGS} ${MPI_LDFLAGS}")

		list(APPEND LIBRARIES ${MPI_C_LIBRARIES})

		set(CMAKE_REQUIRED_FLAGS ${MPI_CFLAGS})   
    
		set(CMAKE_REQUIRED_INCLUDES "${MPI_INCLUDE_DIRS}")
endif()

set(ssw_sources
		main.cc
)
ADD_EXECUTABLE(poisson2d ${ssw_sources})

target_link_libraries(poisson2d ${LIBRARIES} MPI::MPI_CXX)
target_compile_options(poisson2d PRIVATE
     $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
     -Wall -Wextra -pedantic-errors>
     $<$<CXX_COMPILER_ID:MSVC>:
     /W4>)

cmake_print_variables(CMAKE_BUILD_TYPE)

# Tests 
add_test(NAME Compile_and_Run  COMMAND ./poisson2d)
