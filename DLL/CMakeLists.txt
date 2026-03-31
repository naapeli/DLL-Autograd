set(DLL_SOURCES
    pybind_ext.cpp 
    tensor.cpp 
    ops/binary.cpp 
    ops/unary.cpp
    ops/reduce.cpp
    Random/random.cpp
    Random/randomTensor.cpp
)

if(DLL_GPU_AVAILABLE)
    list(APPEND DLL_SOURCES
        gpu_context.cpp
        gpu_buffer.cpp
        ops/gpu_binary.cpp
        ops/gpu_unary.cpp
        ops/gpu_reduce.cpp
    )
endif()

pybind11_add_module(_C ${DLL_SOURCES})

target_compile_features(_C PRIVATE cxx_std_17)
target_include_directories(_C PRIVATE ${CMAKE_SOURCE_DIR}/include /usr/include/x86_64-linux-gnu)
target_link_libraries(_C PRIVATE OpenMP::OpenMP_CXX ${BLAS_LIBRARIES})

target_compile_options(_C PRIVATE -O3 -march=native -ffast-math)

if(DLL_GPU_AVAILABLE)
    target_compile_definitions(_C PRIVATE DLL_GPU_ENABLED)
    target_link_libraries(_C PRIVATE OpenCL::OpenCL ${CLBLAST_LIBRARY})
    if(CLBLAST_INCLUDE_DIR)
        target_include_directories(_C PRIVATE ${CLBLAST_INCLUDE_DIR})
    endif()
    message(STATUS "GPU acceleration: ENABLED (OpenCL + CLBlast)")
else()
    message(STATUS "GPU acceleration: DISABLED")
endif()

add_custom_command(TARGET _C POST_BUILD
    COMMAND python -m pybind11_stubgen _C -o ${CMAKE_CURRENT_SOURCE_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
