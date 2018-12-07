import matplotlib.pyplot as plt

tolerances = []
compression_rates = []

with open('compression_rates.txt') as f:
    for line in f.readlines():
        if line.startswith('# Tolerance'):
            tolerances.append(int(line.split('# Tolerance')[1].split('\n')[0]))

        if line.startswith('Compression rate'):
            compression_rates.append(float(line.split(': ')[1].split('\n')[0]))

plt.plot(tolerances, compression_rates, 'o--', label='ZFP')
plt.xlabel('tolerance')
plt.ylabel('compression rate')
plt.axhline(128 / 17, color='r', label='GZIP')
plt.legend(loc=0)
plt.title('Compression Rate Analysis')
plt.savefig('compression_rates.png', bbox_inches='tight')
