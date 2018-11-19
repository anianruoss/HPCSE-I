include_directories(${PROJECT_SOURCE_DIR})

add_executable(mpi_reduction mpi_reduction/main.cpp)

target_link_libraries(mpi_reduction ${MPI_LIBRARIES})
