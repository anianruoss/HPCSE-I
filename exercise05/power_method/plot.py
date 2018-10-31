import matplotlib.pyplot as plt
import numpy as np

with open('results.txt') as file:
    content = file.readlines()

run_times = []

for idx, line in enumerate(content):
    if line.startswith('RUNTIMES'):
        run_times.append(np.fromstring(content[idx + 1], sep=',')[:-1])

alphas = np.array([1. / 8, 1. / 4, 1. / 2, 1, 3. / 2, 2, 4, 8, 16])

run_times_no_blas = run_times[0]
run_times_blas = run_times[1]
run_times_lapack = run_times[2]

plt.plot(alphas, run_times_no_blas, 'r+:', label='no BLAS')
plt.plot(alphas, run_times_blas, 'b+:', label='BLAS')
plt.plot(alphas, run_times_lapack, 'g+:', label='LAPACK')
plt.xlabel('alphas')
plt.ylabel('run time [s]')
plt.legend()

plt.savefig('run_times.png', bbox_inches='tight')
plt.close()

speedup_blas = np.divide(run_times_no_blas, run_times_blas)
speedup_lapack = np.divide(run_times_blas, run_times_lapack)

plt.plot(alphas, speedup_blas, 'r+:', label='$t_{manual}/t_{power}$')
plt.plot(alphas, speedup_lapack, 'b+:', label='$t_{power}/t_{ev\_full}$')
plt.xlabel('alphas')
plt.ylabel('speedup')
plt.legend()

plt.savefig('speedup.png', bbox_inches='tight')
plt.close()
