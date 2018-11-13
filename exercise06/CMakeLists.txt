include_directories(${PROJECT_SOURCE_DIR})

add_executable(exec_testGrad neural_networks/main_testGrad.cpp)
add_executable(exec_linear neural_networks/main_linear.cpp)
add_executable(exec_nonlinear neural_networks/main_nonlinear.cpp)

target_link_libraries(exec_testGrad blas)
target_link_libraries(exec_linear blas)
target_link_libraries(exec_nonlinear blas)
