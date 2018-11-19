#include <cmath>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <random>

inline long exact(const long N) { return N * (N + 1) / 2; }

void reduce_mpi(long &sum, long &totSum) {
  MPI_Reduce(&sum, &totSum, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
}

// PRE: size is a power of 2 for simplicity
void reduce_manual(int rank, int size, long &sum) {
  // shortcut for single rank
  if (1 < size) {
    int prevSize;
    long otherSum;

    while (1 < size) {
      prevSize = size;
      size /= 2;

      if (size <= rank and rank < prevSize) {
        MPI_Send(&sum, 1, MPI_LONG, rank - size, rank, MPI_COMM_WORLD);
        // printf("size %d: rank %d send to %d \n", size, rank, rank - size);
      } else if (rank < size) {
        MPI_Recv(&otherSum, 1, MPI_LONG, rank + size, rank + size,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        sum += otherSum;
      }
    }
  }
}

int main(int argc, char **argv) {
  const long N = 1000000;

  // Initialize MPI
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // -------------------------
  // Perform the local sum:
  // -------------------------
  long sum = 0;

  // Determine work load per rank
  long N_per_rank = N / size;
  long N_start = rank * N_per_rank;
  long N_end = rank == (size - 1) ? N : (rank + 1) * N_per_rank - 1;

  // N_start + (N_start+1) + ... + (N_start+N_per_rank-1)
  for (long i = N_start; i <= N_end; ++i) {
    sum += i;
  }

  // -------------------------
  // Reduction
  // -------------------------
  long mpi_sum = 0;
  reduce_mpi(sum, mpi_sum);
  reduce_manual(rank, size, sum);

  // -------------------------
  // Print the result
  // -------------------------
  if (rank == 0) {
    std::cout << std::left << std::setw(35)
              << "Final result (exact): " << exact(N) << std::endl;
    std::cout << std::left << std::setw(35)
              << "Final result (MPI Collective): " << mpi_sum << std::endl;
    std::cout << std::left << std::setw(35)
              << "Final result (MPI tree): " << sum << std::endl;
  }

  // Finalize MPI
  MPI_Finalize();

  return 0;
}
