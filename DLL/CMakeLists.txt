pybind11_add_module(_C 
    pybind_ext.cpp 
    tensor.cpp 
    ops/binary.cpp 
    ops/unary.cpp
    ops/reduce.cpp
    Random/mersenneTwister.cpp
    Random/random.cpp
    Random/randomTensor.cpp
)

target_compile_features(_C PRIVATE cxx_std_17)
target_include_directories(_C PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(_C PRIVATE OpenMP::OpenMP_CXX)

if(MINGW)
    target_compile_options(_C PRIVATE -O3 -march=native -ffast-math)
    target_link_options(_C PRIVATE -static -static-libgcc -static-libstdc++)
endif()

get_filename_component(MINGW_BIN_DIR ${CMAKE_CXX_COMPILER} DIRECTORY)

# set(LOCAL_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib")

add_custom_command(TARGET _C POST_BUILD
    # COMMAND ${CMAKE_COMMAND} -E make_directory "${LOCAL_LIB_DIR}"
    # COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR="${MINGW_BIN_DIR}" -DDEST_DIR="${LOCAL_LIB_DIR}" -P "${CMAKE_CURRENT_SOURCE_DIR}/copy_dlls.cmake"
    
    COMMAND python "${CMAKE_CURRENT_SOURCE_DIR}/generate_stub_file.py"
    WORKING_DIRECTORY $<TARGET_FILE_DIR:_C>
)
