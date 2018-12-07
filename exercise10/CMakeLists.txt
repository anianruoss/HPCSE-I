include_directories(${PROJECT_SOURCE_DIR})

add_executable(mpi_compression mpi_compression/mpi_float_compression.cpp)

target_link_libraries(mpi_compression ${MPI_LIBRARIES} ${ZFP_LIBRARIES} -lm -lz)
