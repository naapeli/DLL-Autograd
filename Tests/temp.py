from DLL import rand, randn
import matplotlib.pyplot as plt
from scipy.stats import shapiro, normaltest, kstest


for n in [10, 100, 1000, 10000, 100000]:
    data = randn((n,)).data
    if n < 5000:
        stat, p = shapiro(data)
    else:
        stat, p = normaltest(data)
    print(f"Statistics={stat:.3f}, p-value={p:.3f}")
    if p > 0.05:
        print("Likely Normal (Fail to reject H0)")
    else:
        print("Likely Not Normal (Reject H0)")

for n in [10, 100, 1000, 10000, 100000]:
    data = rand((n,)).data
    stat, p = kstest(data, "uniform", args=(min(data), max(data) - min(data)))
    print(f"Statistics={stat:.3f}, p-value={p:.3f}")
    if p > 0.05:
        print("Likely Uniform")
    else:
        print("Likely Not Uniform")

n = 1000000
x = rand((n,)).data
y = randn((n,)).data

plt.subplot(1, 2, 1)
plt.hist(x, density=True, bins=30)

plt.subplot(1, 2, 2)
plt.hist(y, density=True, bins=30)
plt.show()
