#include <algorithm>
#include <cblas.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

double dnrm2(const size_t N, const double *x) {
  double res = 0.;

  for (size_t i = 0; i < N; ++i) {
    res += x[i] * x[i];
  }

  return std::sqrt(res);
}

double ddot(const size_t N, const double *Aq, const double *q) {
  double res = 0.;

  for (size_t i = 0; i < N; ++i) {
    res += q[i] * Aq[i];
  }

  return res;
}

void dgemv(const size_t N, const double *A, const double *q, double *res) {
  for (size_t i = 0; i < N; ++i) {
    double tmp = 0.;

    for (size_t j = 0; j < N; ++j) {
      tmp += A[i * N + j] * q[j];
    }

    res[i] = tmp;
  }
}

template <class F1, class F2, class F3>
std::pair<size_t, double> powerMethod(const size_t N, double alpha, F1 gemv,
                                      F2 nrm2, F3 dot) {
  auto *q = new double[N];
  std::fill(q, q + N, 0.);
  q[0] = 1.;

  auto *A = new double[N * N];
  std::default_random_engine g(0);
  std::uniform_real_distribution<double> u;

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = i; j < N; ++j) {
      if (i == j) {
        A[(i * N) + j] = alpha * i;
      } else {
        A[(i * N) + j] = A[(j * N) + i] = u(g);
      }
    }
  }

  auto *res = new double[N];
  gemv(N, A, q, res);

  double lambda_old = dot(N, res, q);
  double lambda_new = lambda_old;
  double norm_Aq = nrm2(N, res);

  for (size_t i = 0; i < N; ++i) {
    q[i] = res[i] / norm_Aq;
  }

  size_t k = 0;

  do {
    ++k;

    lambda_old = lambda_new;
    gemv(N, A, q, res);
    lambda_new = dot(N, res, q);

    norm_Aq = nrm2(N, res);

    for (size_t i = 0; i < N; ++i) {
      q[i] = res[i] / norm_Aq;
    }
  } while (std::abs(lambda_new - lambda_old) >= 1e-12);

  delete[] res;
  delete[] A;
  delete[] q;

  return std::make_pair(k, lambda_new);
}

template <class F1, class F2, class F3>
void evaluatePowerMethod(const size_t N, F1 gemv, F2 nrm2, F3 dot) {
  std::vector<double> alphas;
  std::vector<size_t> iterations;
  std::vector<double> eigenvalues;
  std::vector<double> times;

  for (int i = -3; i < 5; ++i) {
    alphas.emplace_back(std::pow(2., i));
    if (i == 0) {
      alphas.emplace_back(3. / 2);
    }
  }

  for (auto &a : alphas) {
    std::cout << "Applying power method with alpha = " << a << std::endl;

    const auto t0 = std::chrono::steady_clock::now();
    auto results = powerMethod(N, a, gemv, nrm2, dot);
    const auto t1 = std::chrono::steady_clock::now();

    std::cout << "Iterations: " << results.first << std::endl;
    std::cout << "Dominant Eigenvalue: " << results.second << std::endl;
    std::cout << std::endl;

    iterations.emplace_back(results.first);
    eigenvalues.emplace_back(results.second);
    times.emplace_back(
        std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0)
            .count());
  }

  long min_index =
      std::distance(iterations.begin(),
                    std::min_element(iterations.begin(), iterations.end()));

  long max_index =
      std::distance(iterations.begin(),
                    std::max_element(iterations.begin(), iterations.end()));

  std::cout << "Fewest iterations: " << iterations[min_index] << std::endl;
  std::cout << "For alpha: " << alphas[min_index] << std::endl;
  std::cout << "With eigenvalue: " << eigenvalues[min_index] << std::endl;
  std::cout << std::endl;
  std::cout << "Most iterations: " << iterations[max_index] << std::endl;
  std::cout << "For alpha: " << alphas[max_index] << std::endl;
  std::cout << "With eigenvalue: " << eigenvalues[max_index] << std::endl;
  std::cout << std::endl;

  std::cout << "RUNTIMES" << std::endl;
  for (const auto &t : times) {
    std::cout << t << ",";
  }
  std::cout << std::endl << std::endl;
}

int main() {
  const size_t N = 1024;

  auto blas_dgemv = std::bind(
      cblas_dgemv, CblasRowMajor, CblasNoTrans, std::placeholders::_1,
      std::placeholders::_1, 1, std::placeholders::_2, std::placeholders::_1,
      std::placeholders::_3, 1, 0, std::placeholders::_4, 1);
  auto blas_dnorm =
      std::bind(cblas_dnrm2, std::placeholders::_1, std::placeholders::_2, 1);
  auto blas_ddot =
      std::bind(cblas_ddot, std::placeholders::_1, std::placeholders::_2, 1,
                std::placeholders::_3, 1);

  std::cout << "POWER METHOD WITHOUT BLAS" << std::endl;
  evaluatePowerMethod(N, dgemv, dnrm2, ddot);

  std::cout << "POWER METHOD WITH BLAS" << std::endl;
  evaluatePowerMethod(N, blas_dgemv, blas_dnorm, blas_ddot);

  return 0;
}
