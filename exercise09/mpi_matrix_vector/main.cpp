#include <algorithm>
#include <cassert>
#include <iostream>
#include <map>
#include <mpi.h>

#include "def.h"
#include "index.h"
#include "io.h"
#include "op.h"

// Adds vectors.
Vect Add(const Vect &a, const Vect &b) {
  auto r = a;
  for (Size i = 0; i < a.v.size(); ++i) {
    r.v[i] += b.v[i];
  }
  return r;
}

// Multiplies vector and scalar.
Vect Mul(const Vect &a, Real k) {
  auto r = a;
  for (auto &e : r.v) {
    e *= k;
  }
  return r;
}

// Multiplies matrix and vector.
Vect Mul(const Matr &a, const Vect &u, MPI_Comm comm) {
  Vect b;
  b.v.resize(L);

  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  std::vector<Size> request_ranks(size, 0);
  std::vector<std::vector<Size>> request_indices(static_cast<Size>(size),
                                                 std::vector<Size>());
  std::vector<std::vector<Real>> request_values(static_cast<Size>(size),
                                                std::vector<Real>());

  // process the element if it is located on the current rank otherwise
  // determine which rank contains it and store the global index
  for (Size locRow = 0; locRow < L; ++locRow) {
    Real value = 0.;

    for (Size k = a.ki[locRow]; k < a.ki[locRow + 1]; ++k) {
      Size valueRank = GlbToRank(a.gjk[k]);

      if (valueRank == static_cast<Size>(rank)) {
        Size locCol = GlbToLoc(a.gjk[k]);
        value += a.a[k] * u.v[locCol];
      } else {
        request_ranks[valueRank] = 1;
        request_indices[valueRank].push_back(a.gjk[k]);
      }
    }
    b.v[locRow] = value;
  }

  // check that no rank wants to send elements to itself
  assert(request_ranks[rank] == 0);
  assert(request_indices[rank].empty());

  // determine how many ranks are going to request elements
  MPI_Allreduce(MPI_IN_PLACE, request_ranks.data(),
                static_cast<int>(request_ranks.size()), MS, MPI_SUM, comm);

  // worst case communication pattern is all-to-all
  Size num_requests = 0;
  MPI_Request requests[3 * size];
  MPI_Status statuses[3 * size];

  // request elements from other ranks
  for (int r = 0; r < size; ++r) {
    if (!request_indices[r].empty()) {
      MPI_Isend(request_indices[r].data(),
                static_cast<int>(request_indices[r].size()), MS, r, 0, comm,
                &requests[num_requests++]);
    }
  }

  // retrieve indices that other ranks need and send the corresponding values
  for (Size i = 0; i < request_ranks[rank]; ++i) {
    int num_elements;
    MPI_Status status;

    MPI_Probe(MPI_ANY_SOURCE, 0, comm, &status);
    MPI_Get_count(&status, MS, &num_elements);

    std::vector<Size> send_indices;
    send_indices.resize(static_cast<Size>(num_elements));

    MPI_Recv(send_indices.data(), static_cast<int>(send_indices.size()), MS,
             status.MPI_SOURCE, 0, comm, MPI_STATUS_IGNORE);

    std::vector<Real> values;
    std::transform(send_indices.begin(), send_indices.end(),
                   std::back_inserter(values),
                   [&u](Size gi) { return u.v[GlbToLoc(gi)]; });

    MPI_Isend(values.data(), static_cast<int>(values.size()), MR,
              status.MPI_SOURCE, 1, comm, &requests[num_requests++]);
  }

  // receive elements from other ranks
  for (int r = 0; r < size; ++r) {
    if (!request_indices[r].empty()) {
      request_values[r].resize(request_indices[r].size());
      MPI_Irecv(request_values[r].data(),
                static_cast<int>(request_values[r].size()), MR, r, 1, comm,
                &requests[r]);
    }
  }

  // after this point all messages have been sent
  MPI_Waitall(static_cast<int>(num_requests), requests, statuses);
  MPI_Barrier(comm);

  // process elements from other ranks
  std::vector<Size> cur_index(static_cast<Size>(size), 0);

  for (Size locRow = 0; locRow < L; ++locRow) {
    Real value = 0.;

    for (Size k = a.ki[locRow]; k < a.ki[locRow + 1]; ++k) {
      Size valueRank = GlbToRank(a.gjk[k]);

      if (valueRank != static_cast<Size>(rank)) {
        value += a.a[k] * request_values[valueRank][cur_index[valueRank]++];
      }
    }
    b.v[locRow] += value;
  }

  return b;
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  MPI_Comm comm = MPI_COMM_WORLD;

  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  // Laplacian
  Matr a = GetLapl(rank);

  // Initial vector
  Vect u;
  for (Size i = 0; i < L; ++i) {
    Size gi = LocToGlb(i, rank);
    auto xy = GlbToCoord(gi);
    Real x = Real(xy[0]) / NX;
    Real y = Real(xy[1]) / NY;
    Real dx = x - 0.5;
    Real dy = y - 0.5;
    Real r = 0.2;
    u.v.push_back(dx * dx + dy * dy < r * r ? 1. : 0.);
  }

  Write(u, comm, "u0");

  const Size nt = 10; // number of time steps
  for (Size t = 0; t < nt; ++t) {
    Vect du = Mul(a, u, comm);

    Real k = 0.25; // scaling, k <= 0.25 required for stability.
    du = Mul(du, k);
    u = Add(u, du);
  }

  Write(u, comm, "u1");

  MPI_Finalize();
}
