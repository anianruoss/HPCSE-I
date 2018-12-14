#include "timer.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>

struct Diagnostics {
  double time;
  double heat;

  Diagnostics(double time, double heat) : time(time), heat(heat) {}
};

class Diffusion2D {
public:
  Diffusion2D(const double D, const double L, const size_t N, const double dt,
              const int rank)
      : D_(D), L_(L), N_(N), dt_(dt), rank_(rank) {
    // Real space grid spacing.
    dr_ = L_ / (N_ - 1);

    // Actual dimension of a row (+2 for the ghost cells).
    real_N_ = N + 2;

    // Total number of cells.
    Ntot_ = (N_ + 2) * (N_ + 2);

    rho_.resize(Ntot_, 0.);
    rho_tmp_.resize(Ntot_, 0.);

    // Initialize field on grid
    initialize_rho();

    R_ = 2. * dr_ * dr_ / dt_;

    // Initialize diagonals of the coefficient matrix A, where Ax=b is the
    // corresponding system to be solved
    const double a = -D / R_;
    const double b = 1 + 2. * D / R_;
    const double c = -D / R_;

    a_.resize(real_N_, a);
    b_.resize(real_N_, b);
    c_.resize(real_N_, c);

    a_[0] = a_[1] = a_[real_N_ - 1] = 0.;
    c_[0] = c_[real_N_ - 2] = c_[real_N_ - 1] = 0.;
    b_[0] = b_[real_N_ - 1] = 1.;

    // pre-compute first step of Thomas algorithm
    for (size_t i = 1; i < real_N_; ++i) {
      b_[i] -= c_[i - 1] * a_[i] / b_[i - 1];
      assert(b_[i] != 0);
    }
  }

  void advance() {
    // ADI Step 1: Update rows at half time step
    // Solve implicit system with Thomas algorithm
    std::vector<double> v(real_N_, 0.);

    for (size_t j = 1; j < real_N_ - 1; ++j) {
      v[0] = v[real_N_ - 1] = 0.;

      for (size_t i = 1; i < real_N_ - 1; ++i) {
        v[i] =
            D_ * (rho_[i * real_N_ + j + 1] + rho_[i * real_N_ + j - 1]) / R_ +
            (1. - 2. * D_ / R_) * rho_[i * real_N_ + j];
      }

      // first step of Thomas algorithm
      for (size_t i = 1; i < real_N_; ++i) {
        v[i] -= v[i - 1] * a_[i] / b_[i - 1];
      }

      // second step of Thomas algorithm
      rho_tmp_[(real_N_ - 1) * real_N_ + j] = v[real_N_ - 1] / b_[real_N_ - 1];

      for (int i = static_cast<int>(real_N_) - 2; i >= 0; --i) {
        rho_tmp_[i * real_N_ + j] =
            (v[i] - c_[i] * rho_tmp_[(i + 1) * real_N_ + j]) / b_[i];
      }
    }

    std::copy(rho_tmp_.begin(), rho_tmp_.end(), rho_.begin());

    // ADI Step 2: Update columns at full time step
    // Solve implicit system with Thomas algorithm
    for (size_t i = 1; i < real_N_ - 1; ++i) {
      v[0] = v[real_N_ - 1] = 0.;

      for (size_t j = 1; j < real_N_ - 1; ++j) {
        v[j] = D_ *
                   (rho_[(i + 1) * real_N_ + j] + rho_[(i - 1) * real_N_ + j]) /
                   R_ +
               (1. - 2. * D_ / R_) * rho_[i * real_N_ + j];
      }

      // first step of Thomas algorithm
      for (size_t j = 1; j < real_N_; ++j) {
        v[j] -= v[j - 1] * a_[j] / b_[j - 1];
      }

      // second step of Thomas algorithm
      rho_tmp_[i * real_N_ + (real_N_ - 1)] = v[real_N_ - 1] / b_[real_N_ - 1];

      for (int j = static_cast<int>(real_N_) - 2; j >= 0; --j) {
        rho_tmp_[i * real_N_ + j] =
            (v[j] - c_[j] * rho_tmp_[i * real_N_ + (j + 1)]) / b_[j];
      }
    }

    std::copy(rho_tmp_.begin(), rho_tmp_.end(), rho_.begin());
  }

  void compute_diagnostics(const double t) {
    double heat = 0.0;
    for (size_t i = 1; i <= N_; ++i)
      for (size_t j = 1; j <= N_; ++j)
        heat += dr_ * dr_ * rho_[i * real_N_ + j];

#if DEBUG
    std::cout << "t = " << t << " heat = " << heat << '\n';
#endif
    diag_.emplace_back(Diagnostics(t, heat));
  }

  void write_diagnostics(const std::string &filename) const {
    std::ofstream out_file(filename, std::ios::out);
    for (const Diagnostics &d : diag_)
      out_file << d.time << '\t' << d.heat << '\n';
    out_file.close();
  }

private:
  /* Initialize rho(x, y, t=0) */
  void initialize_rho() {
    double bound = 0.25 * L_;
    // Initialize rho based on the prescribed initial conditions
    for (size_t i = 1; i <= N_; ++i) {
      for (size_t j = 1; j <= N_; ++j) {
        if ((std::abs((i - 1.) * dr_ - L_ / 2.) < bound) &&
            (std::abs((j - 1.) * dr_ - L_ / 2.) < bound)) {
          rho_[i * real_N_ + j] = rho_tmp_[i * real_N_ + j] = 1.;
        }
      }
    }
  }

  double D_, L_;
  size_t N_, Ntot_, real_N_;
  double dr_, dt_;
  double R_;
  int rank_;
  std::vector<double> rho_, rho_tmp_;
  std::vector<Diagnostics> diag_;
  std::vector<double> a_, b_, c_;
};

int main(int argc, char *argv[]) {
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0] << " D L N dt\n";
    return 1;
  }

#pragma omp parallel
  {
#pragma omp master
    std::cout << "Running with " << omp_get_num_threads() << " threads\n";
  }

  const double D = std::stod(argv[1]);  // diffusion constant
  const double L = std::stod(argv[2]);  // domain side size
  const size_t N = std::stoul(argv[3]); // number of grid points per dimension
  const double dt = std::stod(argv[4]); // time step

  Diffusion2D system(D, L, N, dt, 0);

  timer t;
  t.start();
  for (int step = 0; step < 10000; ++step) {
    system.advance();
#ifndef _PERF_
    system.compute_diagnostics(dt * step);
#endif
  }
  t.stop();

  std::cout << "Timing: " << N << ' ' << t.get_timing() << '\n';

#ifndef _PERF_
  system.write_diagnostics("diagnostics_openmp.dat");
#endif

  return 0;
}
