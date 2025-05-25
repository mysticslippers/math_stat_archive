import math

import pandas as pd
from scipy.stats import t as t_dist, spearmanr

df = pd.read_csv('kc_house_data.csv')
df.head()

prices = df["price"]

# Входные данные
x = prices
y = df["sqft_living"]
n = len(x)

# Средние значения
x_bar = sum(x) / n
y_bar = sum(y) / n

# Числитель и знаменатель для корреляции Пирсона
numerator = sum((xi - x_bar) * (yi - y_bar) for xi, yi in zip(x, y))
denominator_x = sum((xi - x_bar) ** 2 for xi in x)
denominator_y = sum((yi - y_bar) ** 2 for yi in y)

# r — коэффициент корреляции Пирсона
r = numerator / math.sqrt(denominator_x * denominator_y)

# t-статистика
t_stat = r * math.sqrt(n - 2) / math.sqrt(1 - r ** 2)

# Параметры критерия
alpha = 0.05
degree_freedom = n - 2

# Критическое значение t (двусторонний критерий)
t_crit = t_dist.ppf(1 - alpha / 2, degree_freedom)

# Двусторонний p-value
p_value = 2 * (1 - t_dist.cdf(abs(t_stat), degree_freedom))

# Вывод
print("Гипотезы:")
print("H0: rho = 0 (нет корреляции)")
print("H1: rho ≠ 0 (есть корреляция)\n")

print(f"Коэффициент корреляции r: {r:.4f}")
print(f"t-статистика: {t_stat:.4f}")
print(f"Критическое значение t (двусторонний тест, alpha = 0.05): ±{t_crit:.4f}")
print(f"p-value (двусторонний тест): {p_value:.6f}")

print("\nРешение:")
if abs(t_stat) > t_crit:
    print("Отклоняем H0 — есть статистически значимая корреляция.")
else:
    print("Не отклоняем H0 — доказательств корреляции недостаточно.")

print("Оценка коэффициента Спирмена.")
# Вычисляем коэффициент Спирмена и p-value
rho_s, p_value = spearmanr(x, y, alternative="two-sided")

# t-статистика (приближённая оценка)
t_stat = rho_s * math.sqrt((n - 2) / (1 - rho_s ** 2))

# Критическое значение t (двусторонний тест)
alpha = 0.05
degree_freedom = n - 2
t_crit = t_dist.ppf(1 - alpha / 2, degree_freedom)

# Вывод
print("Гипотезы:")
print("H0: rho_s = 0 (нет корреляции)")
print("H1: rho_s ≠ 0 (есть корреляция)\n")

print(f"Коэффициент Спирмена rho_s: {rho_s:.4f}")
print(f"t-статистика (приближение): {t_stat:.4f}")
print(f"Критическое значение t (двусторонний тест, alpha = 0.05): ±{t_crit:.4f}")
print(f"p-value (двусторонний тест): {p_value:.6f}")

print("\nРешение:")
if abs(t_stat) > t_crit:
    print("Отклоняем H0 — есть статистически значимая корреляция (по Спирмену).")
else:
    print("Не отклоняем H0 — доказательств корреляции недостаточно.")