include_directories(${PROJECT_SOURCE_DIR})

add_executable(mpi_matrix_vector mpi_matrix_vector/main.cpp)

target_link_libraries(mpi_matrix_vector ${MPI_LIBRARIES})
