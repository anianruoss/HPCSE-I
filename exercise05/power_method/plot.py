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

plt.plot(alphas, run_times_no_blas, 'r--', label='no BLAS')
plt.plot(alphas, run_times_blas, 'b--', label='BLAS')
plt.legend()

plt.savefig('run_times.png', bbox_inches='tight')
plt.close()

speedup = np.divide(run_times_no_blas, run_times_blas)

plt.plot(alphas, speedup)
plt.savefig('speedup.png', bbox_inches='tight')
plt.close()
