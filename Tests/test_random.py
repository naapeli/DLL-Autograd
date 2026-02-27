import pytest
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, normaltest, kstest, uniform, norm
from DLL import rand, randn


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_normality(n):
    data = randn((n,)).data
    if n < 1000: _, p = shapiro(data)
    else: _, p = normaltest(data)

    assert p > 0.05, f"n={n} failed normality test with p={p:.4f}"

@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_uniformity(n):
    data = rand((n,)).data    
    _, p = kstest(data, "uniform", args=(0, 1))
    
    assert p > 0.05, f"n={n} failed uniformity test with p={p:.4f}"


def plot_distributions():
    n = 1000000
    x = rand((n,)).data
    y = randn((n,)).data

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    x_pdf_range = np.linspace(-0.1, 1.1, 1000)
    plt.plot(x_pdf_range, uniform.pdf(x_pdf_range, 0, 1), 'r-', lw=2, label='True PDF')
    plt.hist(x, density=True, bins=50, alpha=0.7, color='skyblue')
    plt.title(f"Uniform Distribution (n={n})")
    plt.legend()

    plt.subplot(1, 2, 2)
    y_pdf_range = np.linspace(-4, 4, 1000)
    plt.plot(y_pdf_range, norm.pdf(y_pdf_range, 0, 1), 'b-', lw=2, label='True PDF')
    plt.hist(y, density=True, bins=50, alpha=0.7, color='salmon')
    plt.title(f"Normal Distribution (n={n})")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_distributions()
