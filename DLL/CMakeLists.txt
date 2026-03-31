pybind11_add_module(_C 
    pybind_ext.cpp 
    tensor.cpp 
    ops/binary.cpp 
    ops/unary.cpp
    ops/reduce.cpp
    Random/random.cpp
    Random/randomTensor.cpp
)

target_compile_features(_C PRIVATE cxx_std_17)
target_include_directories(_C PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(_C PRIVATE OpenMP::OpenMP_CXX)

if(MSVC)
    target_compile_options(_C PRIVATE /O2 /fp:fast /arch:AVX2)
else()
    target_compile_options(_C PRIVATE -O3 -march=native -ffast-math)
endif()

if(WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND NOT MSVC)
    target_link_options(_C PRIVATE -static -static-libgcc -static-libstdc++)
endif()

add_custom_command(TARGET _C POST_BUILD
    COMMAND python -m pybind11_stubgen _C -o ${CMAKE_CURRENT_SOURCE_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
