import pandas as pd
from scipy import stats
import numpy as np

data = pd.read_csv("kc_house_data.csv")

prices = data['price']

mu, std = np.mean(prices), np.std(prices)

D, p_value = stats.kstest(prices, 'norm', args=(mu, std))

alpha = 0.05

critical_value = stats.norm.ppf(1 - alpha/2)

print(f"Статистика теста Колмогорова: {D}")
print(f"p-value: {p_value}")
print(f"Критическое значение: {critical_value}\n")

shapiro_stat, shapiro_pvalue = stats.shapiro(prices)

print(f"Статистика теста Шапиро: {shapiro_stat}")
print(f"p-value для теста Шапиро: {shapiro_pvalue}\n")

if p_value < alpha:
    print("Результаты теста Колмогорова:")
    print("p-value меньше уровня значимости. Мы отвергаем нулевую гипотезу о нормальности распределения.\n")
else:
    print("Результаты теста Колмогорова:")
    print("p-value больше или равно уровню значимости. У нас нет оснований отвергать нулевую гипотезу о нормальности распределения.\n")

if shapiro_pvalue < alpha:
    print("Результаты теста Шапиро-Уилка:")
    print("p-value меньше уровня значимости. Мы отвергаем нулевую гипотезу о нормальности распределения.\n")
else:
    print("Результаты теста Шапиро-Уилк:")
    print("p-value больше или равно уровню значимости. У нас нет оснований отвергать нулевую гипотезу о нормальности распределения.\n")

if p_value < alpha or shapiro_pvalue < alpha:
    print("В целом, данные, судя по обоим тестам, не являются нормально распределенными.")
else:
    print("В целом, данные, судя по обоим тестам, могут быть нормально распределены.")