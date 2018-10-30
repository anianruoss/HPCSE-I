include_directories(${PROJECT_SOURCE_DIR}})

add_executable(power_method power_method/main.cpp)
target_link_libraries(power_method blas lapack)