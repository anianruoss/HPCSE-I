include_directories(${PROJECT_SOURCE_DIR})

add_executable(testGrad convolutional_neural_networks/main_testGrad.cpp)
add_executable(classify convolutional_neural_networks/main_classify.cpp)
add_executable(classify_improved
        convolutional_neural_networks/main_classify_improved.cpp)

target_link_libraries(testGrad blas)
target_link_libraries(classify blas)
target_link_libraries(classify_improved blas)
