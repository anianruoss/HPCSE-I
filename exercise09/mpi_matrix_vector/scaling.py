import matplotlib.pyplot as plt
import numpy as np

ranks = np.array([1, 4, 9, 16, 25, 36])
times = np.array([1.839, 2.125, 2.169, 2.750, 4.014, 6.313])
serial = times[0]

plt.plot(ranks, ranks * serial / times, 'o-')
plt.plot([1, 36], [1, 36], 'k-')
plt.xlim(left=1, right=36)
plt.ylim(bottom=1, top=36)
plt.xlabel('# ranks')
plt.ylabel(r'Speedup:  $p*\frac{t_1}{t_p}$')
plt.title('Weak Scaling Analysis')
plt.savefig('scaling.png', bbox_inches='tight')
