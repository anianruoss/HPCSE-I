#include "timer.h"
#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <iostream>


using numeric_t = double;

numeric_t flushCache() {
    const size_t N = 1000;
    const Eigen::MatrixXd A = Eigen::MatrixXd::Random(N, N);
    const Eigen::VectorXd x = Eigen::MatrixXd::Random(N, 1);

    return x.transpose() * A * x;
}

int main() {
    const numeric_t alpha = 1e-4;
    const size_t L = 1000;
    const size_t T = 1;
    const size_t N = 1000000;
    const numeric_t tau = 0.001;

    assert(tau < std::pow((L / static_cast<numeric_t>(N)), 2) / (2. * alpha));

    auto u0 = [](numeric_t x) {
        return std::sin(2. * M_PI * x / L);
    };
    Eigen::MatrixXd U = Eigen::MatrixXd::Zero(N, T / tau);

    Timer timer;

    for (size_t lap = 0; lap < 10; ++lap) {
        std::cout << "starting lap: " << lap << std::endl;
        timer.start();

        const Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(N, 0, L);
        U.col(0) = x.unaryExpr(u0);

        for (size_t t = 0; t < ((T / tau) - 1); ++t) {
            for (size_t i = 0; i < N; ++i) {
                U(i, t + 1) = U(i, t) + (tau * alpha / (x(1) * x(1))) * (
                        U(((i - 1) + N) % N, t) - 2. * U(i, t) +
                        U((i + 1) % N, t)
                );
            }
        }

        timer.lap();

        std::cout << "flushing cache: " << flushCache() << std::endl;
    }

    std::cout << "Average runtime: " << timer.mean() << "s" << std::endl;
    std::cout << "Minimal runtime: " << timer.min() << "s" << std::endl;

    return 0;
}
