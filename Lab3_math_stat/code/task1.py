from math import sqrt

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm, kstest, anderson

df = pd.read_csv('kc_house_data.csv')
df.head()

prices = df["price"]
# Построение гистограммы
plt.figure(figsize=(10, 6))
plt.hist(prices, bins=50)
plt.title('Распределение цен на дома')
plt.xlabel('Цена')
plt.ylabel('Количество домов')
plt.grid(True)
plt.show()

mu_hat = prices.mean()
sigma_hat = prices.std(ddof=1) #несмещённое стандартное отклонение
print(mu_hat, sigma_hat)

def ks_p_value(D, n, terms=100):
    """Асимптотическая аппроксимация p-value для К-С теста"""
    lambda_val = sqrt(n) * D
    s = 0.0
    for k in range(1, terms + 1):
        s += (-1)**(k - 1) * np.exp(-2 * (k * lambda_val)**2)
    return 2 * s

# Эмпирическая и теоретическая CDF
prices_sorted = np.sort(prices)
n = len(prices_sorted)

empirical_cdf = np.arange(1, n + 1) / n
theoretical_cdf = norm.cdf(prices_sorted, loc=mu_hat, scale=sigma_hat)

# Статистика критерия
D_statistic = np.max(np.abs(empirical_cdf - theoretical_cdf))

alpha = 0.05
# Критическое значение (табличное для alpha = 0.05)
D_critical = 1.36 / sqrt(n)

# Вычисляем  p-value
p_value = ks_p_value(D_statistic, n)


# Вывод
print(f"Оценка параметров: mu = {mu_hat:.2f}, sigma = {sigma_hat:.2f}")
print(f"Статистика D = {D_statistic:.4f}")
print(f"Критическое значение D_crit (alpha={alpha}) = {D_critical:.4f}")
print(f"p-value (аппроксимация) ≈ {p_value:.6f}")

if D_statistic > D_critical:
    print("H0 отвергается: распределение цен не нормальное.")
else:
    print("H0 не отвергается: нормальность распределения допустима.")

print("Проверим scipy: ")
statistic, p_value = kstest(prices, 'norm', args=(mu_hat, sigma_hat))

# Вывод результатов
print(f"KS-статистика: {statistic:.4f}")
print(f"p-value: {p_value:.6f}")

# Критерий Андерсона-Дарлинга для нормального распределения с значения
result = anderson(prices, dist='norm')

print("Статистика теста A²:", result.statistic)
print("Критические значения:", result.critical_values)
print("Уровни значимости:", result.significance_level)

# Интерпретация
for sl, cv in zip(result.significance_level, result.critical_values):
    if result.statistic < cv:
        print(f"На уровне значимости {sl}%: гипотеза о нормальности НЕ отвергается")
    else:
        print(f"На уровне значимости {sl}%: гипотеза о нормальности ОТВЕРГАЕТСЯ")