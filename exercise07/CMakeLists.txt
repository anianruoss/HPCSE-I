include_directories(${PROJECT_SOURCE_DIR})

add_executable(testGrad convolutional_neural_networks/main_testGrad.cpp)
add_executable(classify convolutional_neural_networks/main_classify.cpp)

target_link_libraries(testGrad blas)
target_link_libraries(classify blas)