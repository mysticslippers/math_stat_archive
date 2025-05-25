from math import sqrt

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import ks_2samp

df = pd.read_csv('kc_house_data.csv')
df.head()

# Определим порог между старым и новым фондом
year_threshold = 1980

# Разделим выборку по году постройки
old_houses = df[df['yr_built'] < year_threshold]['price'].dropna()
new_houses = df[df['yr_built'] >= year_threshold]['price'].dropna()

print(f"Старых домов: {len(old_houses)}")
print(f"Новых домов: {len(new_houses)}")

def ks_p_value(D, n, terms=100):
    """Асимптотическая аппроксимация p-value для К-С теста"""
    lambda_val = sqrt(n) * D
    s = 0.0
    for k in range(1, terms + 1):
        s += (-1)**(k - 1) * np.exp(-2 * (k * lambda_val)**2)
    return 2 * s
all_prices = np.sort(np.concatenate((old_houses.values, new_houses.values)))

# Для каждого значения считаем эмпирические функции распределения (ECDF)
def ecdf(data, x):
    return np.searchsorted(np.sort(data), x, side='right') / len(data)

# Найдём максимум абсолютного отклонения между ECDF
d_statistic = 0
for x in all_prices:
    d = abs(ecdf(old_houses, x) - ecdf(new_houses, x))
    if d > d_statistic:
        d_statistic = d

# Функция ECDF
def ecdf(data, x):
    return np.searchsorted(np.sort(data), x, side='right') / len(data)

# Вычисляем максимальное отклонение
d_statistic = 0
for x in all_prices:
    d = abs(ecdf(old_houses, x) - ecdf(new_houses, x))
    if d > d_statistic:
        d_statistic = d

# Эффективный размер выборки
n1 = len(old_houses)
n2 = len(new_houses)
n_eff = (n1 * n2) / (n1 + n2)

# Вычисляем p-value с помощью твоей функции
p_value = ks_p_value(d_statistic, n_eff)

# Ограничим p-value сверху 1
p_value = min(1, p_value)

# --- Вывод результатов ---
print("\n Результаты KS-теста ")
print(f"KS-статистика: {d_statistic:.5f}")
print(f"Эффективный размер n: {n_eff:.2f}")
print(f"p-значение: {p_value:.5f}")

# Интерпретация
if p_value < 0.05:
    print("\n→ Распределения цен статистически различаются (отвергаем H₀)")
else:
    print("\n→ Нет оснований отвергать равенство распределений (не отвергаем H₀)")

print("Проверим с помощью библиотеки: ")
ks_stat, ks_p = ks_2samp(old_houses, new_houses)
print(ks_stat)

df0 = df[['price', 'yr_built']]

# Разделяем на старый и новый фонд
old_fund = df0[df0['yr_built'] <= 1980]['price']
new_fund = df0[df0['yr_built'] > 1980]['price']

# Считаем квартильные границы по всей выборке цен
all_prices = df0['price']
q1, q2, q3 = np.percentile(all_prices, [25, 50, 75])

# Присваиваем номер квартиля каждой цене
def get_quartile(price):
    if price <= q1:
        return 1
    elif price <= q2:
        return 2
    elif price <= q3:
        return 3
    else:
        return 4

old_quartiles = [get_quartile(p) for p in old_fund]
new_quartiles = [get_quartile(p) for p in new_fund]

# Подсчёт наблюдаемых частот (частоты попаданий в каждый квартиль)
obs_old = [old_quartiles.count(k) for k in [1, 2, 3, 4]]
obs_new = [new_quartiles.count(k) for k in [1, 2, 3, 4]]

observed = [obs_old, obs_new]

# Суммы по строкам и столбцам
row_sums = [sum(row) for row in observed]  # всего в старом и новом фонде
col_sums = [obs_old[i] + obs_new[i] for i in range(4)]  # по квартилям
total = sum(row_sums)

# Расчёт статистики χ² вручную
chi2 = 0.0
for i in range(2):  # по строкам (группы)
    for j in range(4):  # по столбцам (квартиль)
        expected = row_sums[i] * col_sums[j] / total
        chi2 += (observed[i][j] - expected) ** 2 / expected

# Степени свободы
df = (2 - 1) * (4 - 1)

print("Хи-квадрат статистика:", round(chi2, 2))
print("Степени свободы:", df)

critical_05 = 7.815  # хи-квадрат критическое значение при df=3 и α=0.05
if chi2 > critical_05:
    print("Отвергаем H0: распределения цен отличаются.")
else:
    print("Нет оснований отвергать H0: распределения цен похожи.")

plt.figure(figsize=(12, 6))
sns.kdeplot(old_houses, label='Старые дома (<1980)', fill=True)
sns.kdeplot(new_houses, label='Новые дома (≥1980)', fill=True)
plt.title('Сравнение распределения цен для старых и новых домов')
plt.xlabel('Цена')
plt.ylabel('Плотность')
plt.legend()
plt.grid(True)
plt.show()