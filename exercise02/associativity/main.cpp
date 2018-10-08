#include <chrono>
#include <cstdio>


using interval_t = std::chrono::duration<double>;

void measure_flops(int N, int K) {
    volatile double *buf = new volatile double[N * K];
    for (int i = 0; i < N * K; ++i) {
        buf[i] = 0;
    }

    int repeat = 500 / K;

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; ++i) {
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) {
                ++buf[(k * N) + n];
            }
        }
    }
    const auto t1 = std::chrono::steady_clock::now();

    delete[] buf;

    // Report.
    double time = std::chrono::duration_cast<interval_t>(t1 - t0).count();
    double flops = (double) repeat * N * K / time;
    printf("%d  %2d  %.4lf\n", N, K, flops * 1e-9);
    fflush(stdout);
}

void run(int N) {
    printf("      N   K  GFLOPS\n");
    for (int K = 1; K <= 40; ++K)
        measure_flops(N, K);
    printf("\n\n");
}

int main() {
    // Array size. Must be a multiple of a large power of two.
    const int N = 1 << 20;

    // Power of two size --> bad.
    run(N);

    // Non-power-of-two size --> better.
    run(N + 64 / sizeof(double));

    return 0;
}

