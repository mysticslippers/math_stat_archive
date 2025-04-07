import numpy as np
from scipy.stats import f

def simulate_coverage(n):
  F_low = f.ppf(0.05 / 2, n, n)
  F_high = f.ppf(1 - 0.05 / 2, n, n)

  X1 = np.random.normal(0, np.sqrt(2), (1000, n))
  X2 = np.random.normal(0, np.sqrt(1), (1000, n))

  sum_X1_sq = np.sum(X1**2, axis=1)
  sum_X2_sq = np.sum(X2**2, axis=1)

  tau_hat = (sum_X1_sq / n) / (sum_X2_sq / n)

  ci_low = tau_hat * F_low
  ci_high = tau_hat * F_high

  coverages = np.sum((ci_low <= 2) & (ci_high >= 2))

  return coverages / 1000

print(f"Покрытие для малого объёма 25: {simulate_coverage(25):.3f}")
print(f"Покрытие для большого объёма 10000: {simulate_coverage(10000):.3f}")
