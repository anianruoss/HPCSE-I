import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('diagnostics_openmp.dat')

plt.plot(data[:, 0], data[:, 1])
plt.xlabel('time')
plt.ylabel('total heat')
plt.title('2D Diffusion with ADI Scheme')
plt.savefig('heat_flow.png', bbox_inches='tight')
