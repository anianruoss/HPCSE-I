#include <iostream>
#include <algorithm>
#include <string>
#include <fstream>
#include <cassert>
#include <vector>
#include <cmath>
#include "timer.hpp"

// TODO
// Include OpenMP header
// ...



struct Diagnostics
{
    double time;
    double heat;

    Diagnostics(double time, double heat) : time(time), heat(heat) {}
};

class Diffusion2D
{
public:
    Diffusion2D(const double D,
                const double L,
                const int N,
                const double dt,
                const int rank)
            : D_(D), L_(L), N_(N), dt_(dt), rank_(rank)
    {
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

        R_ = 2.*dr_*dr_ / dt_;
        // TODO:
        // Initialize diagonals of the coefficient
        // matrix A, where Ax=b is the corresponding
        // system to be solved
        // ...
    }


    void advance()
    {
        // TODO:
        // Implement the ADI scheme for diffusion
        // and parallelize with OpenMP
        // ...


        // ADI Step 1: Update rows at half timestep
        // Solve implicit system with Thomas algorithm
        // ...


        // ADI: Step 2: Update columns at full timestep
        // Solve implicit system with Thomas algorithm
        // ...

    }


    void compute_diagnostics(const double t)
    {
        double heat = 0.0;
        for (int i = 1; i <= N_; ++i)
            for (int j = 1; j <= N_; ++j)
                heat += dr_ * dr_ * rho_[i * real_N_ + j];

#if DEBUG
        std::cout << "t = " << t << " heat = " << heat << '\n';
#endif
        diag_.emplace_back(Diagnostics(t, heat));
    }


    void write_diagnostics(const std::string &filename) const
    {
        std::ofstream out_file(filename, std::ios::out);
        for (const Diagnostics &d : diag_)
            out_file << d.time << '\t' << d.heat << '\n';
        out_file.close();
    }


private:

    void initialize_rho()
    {
        /* Initialize rho(x, y, t=0) */

        double bound = 0.25 * L_;

        // TODO:
        // Initialize field rho based on the
        // prescribed initial conditions
        // and parallelize with OpenMP
        // ...
        
        for (int i = 1; i <= N_; ++i) {
            for (int j = 1; j <= N_; ++j) {
                
            // ...

            }
        }
  
    }



    double D_, L_;
    int N_, Ntot_, real_N_;
    double dr_, dt_;
    double R_;
    int rank_;
    std::vector<double> rho_, rho_tmp_;
    std::vector<Diagnostics> diag_;
    std::vector<double> a_, b_, c_;
};



int main(int argc, char* argv[])
{
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " D L N dt\n";
        return 1;
    }

#pragma omp parallel
    {
#pragma omp master
        std::cout << "Running with " << omp_get_num_threads() << " threads\n";
    }

    const double D = std::stod(argv[1]);  //diffusion constant
    const double L = std::stod(argv[2]);  //domain side size
    const int N = std::stoul(argv[3]);    //number of grid points per dimension
    const double dt = std::stod(argv[4]); //timestep

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
