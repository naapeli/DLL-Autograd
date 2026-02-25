# Require a modern version of CMake
cmake_minimum_required(VERSION 3.15...3.28)

# Name your C++ project
project(MyTensorEngine)

# --- NEW CODE START ---
# Ask Python where it installed pybind11's CMake configuration files
execute_process(
    COMMAND python -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
# --- NEW CODE END ---

# Now CMake knows exactly where to look!
find_package(pybind11 CONFIG REQUIRED)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/DLL)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/DLL)

# Create the Python module.
add_subdirectory(DLL)

# Require C++17
target_compile_features(_c_tensor PRIVATE cxx_std_17)

if(MINGW)
    target_link_options(_c_tensor PRIVATE "-static")  # solves ImportError: DLL load failed while importing _c_tensor
endif()
