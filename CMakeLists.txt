cmake_minimum_required(VERSION 3.15...3.28)

project(DLLAutograd)

execute_process(
    COMMAND python -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

option(DLL_USE_GPU "Enable OpenCL GPU acceleration" ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)

if(DLL_USE_GPU)
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
        message(STATUS "OpenCL found: ${OpenCL_LIBRARIES}")
        # Look for CLBlast
        find_library(CLBLAST_LIBRARY clblast)
        find_path(CLBLAST_INCLUDE_DIR clblast.h PATH_SUFFIXES include)
        if(CLBLAST_LIBRARY)
            message(STATUS "CLBlast found: ${CLBLAST_LIBRARY}")
            set(DLL_GPU_AVAILABLE ON)
        else()
            message(WARNING "CLBlast not found — building without GPU matmul. Install with: sudo apt install libclblast-dev")
            set(DLL_GPU_AVAILABLE OFF)
        endif()
    else()
        message(WARNING "OpenCL not found — building CPU-only. Install with: sudo apt install opencl-headers ocl-icd-opencl-dev")
        set(DLL_GPU_AVAILABLE OFF)
    endif()
else()
    set(DLL_GPU_AVAILABLE OFF)
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/DLL)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/DLL)

add_subdirectory(DLL)
