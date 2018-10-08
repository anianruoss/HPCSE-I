#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

const int MIN_N = 1000 / sizeof(int);      // From 1 KB
const int MAX_N = 20000000 / sizeof(int);  // to 20 MB.
const int NUM_SAMPLES = 100;
const int M = 100000000;    // Operations per sample.
int a[MAX_N];               // Permutation array.


void sattolo(int *p, int N) {
    /*
     * Generate a random single-cycle permutation using Satollo's algorithm.
     * https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle#Sattolo's_algorithm
     */
    for (int i = 0; i < N; ++i)
        p[i] = i;
    for (int i = 0; i < N - 2; ++i)
        std::swap(p[i], p[i + 1 + rand() % (N - i - 1)]);
}

double measure(int N, int mode) {
    if (mode == 0) {
        sattolo(a, N);
    } else if (mode == 1) {
        volatile int k = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < M; ++i)
            k = a[i];
        const auto t1 = std::chrono::steady_clock::now();

        return (t1 - t0).count();

    } else if (mode == 2) {
        // TODO: Question 1d: Initialize the permutation such that k jumps by 64 bytes (cyclically).

    }

    volatile int k = 0;

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < M; ++i)
        k = a[k];
    const auto t1 = std::chrono::steady_clock::now();

    return (t1 - t0).count();
}

void run_mode(int mode) {
    /*
     * Run the measurement for many different values of N and output in a
     * format compatible with the plotting script.
     */
    printf("%9s  %9s  %7s  %7s\n", "N", "size[kB]", "t[s]",
           "op_per_sec[10^9]");
    for (int i = 0; i < NUM_SAMPLES; ++i) {
        // Generate N in a logarithmic scale.
        int N = (int) (MIN_N * std::pow((double) MAX_N / MIN_N,
                                        (double) i / (NUM_SAMPLES - 1)));
        double t = measure(N, mode);
        printf("%9d  %9.1lf  %7.5lf  %7.6lf\n",
               N, N * sizeof(int) / 1024., t, M / t * 1e-9);
        fflush(stdout);
    }
    printf("\n\n");
}

int main() {
    run_mode(0);   // Random.
    run_mode(1);   // Sequential (jump by sizeof(int) bytes).

    // TODO: Enable for Question 1d:
    // run_mode(2);   // Sequential (jump by cache line size, i.e. 64 bytes).

    return 0;
}

