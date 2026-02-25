cmake_minimum_required(VERSION 3.15...3.28)

project(DLLAutograd)

execute_process(
    COMMAND python -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 CONFIG REQUIRED)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/DLL)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/DLL)

add_subdirectory(DLL)

target_compile_features(_c_tensor PRIVATE cxx_std_17)

target_include_directories(_c_tensor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(MINGW)
    target_link_options(_c_tensor PRIVATE "-static") 
endif()
