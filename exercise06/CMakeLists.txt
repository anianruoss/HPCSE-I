include_directories(${PROJECT_SOURCE_DIR})

add_executable(exec_testGrad main_testGrad.cpp)
add_executable(exec_linear main_linear.cpp)
add_executable(exec_nonlinear main_nonlinear.cpp)

target_link_libraries(exec_testGrad blas)
target_link_libraries(exec_linear blas)
target_link_libraries(exec_linear blas)
