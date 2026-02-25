pybind11_add_module(_c_tensor 
    pybind_ext.cpp 
    tensor.cpp 
    ops/binary.cpp 
    ops/unary.cpp
    ops/reduce.cpp
)

add_custom_command(TARGET _c_tensor POST_BUILD
    COMMAND python -m pybind11_stubgen _c_tensor -o ${CMAKE_CURRENT_SOURCE_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
