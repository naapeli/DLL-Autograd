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
target_include_directories(_C PRIVATE ${CMAKE_SOURCE_DIR}/include /usr/include/x86_64-linux-gnu)
target_link_libraries(_C PRIVATE OpenMP::OpenMP_CXX ${BLAS_LIBRARIES})

target_compile_options(_C PRIVATE -O3 -march=native -ffast-math)

add_custom_command(TARGET _C POST_BUILD
    COMMAND python -m pybind11_stubgen _C -o ${CMAKE_CURRENT_SOURCE_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
