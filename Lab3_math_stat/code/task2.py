import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

data = pd.read_csv("kc_house_data.csv")

threshold_year = 1993

old_houses = data[data['yr_built'] < threshold_year]['price']
new_houses = data[data['yr_built'] >= threshold_year]['price']

plt.figure(figsize=(10, 6))
sns.kdeplot(old_houses, label='Старые дома', color='blue')
sns.kdeplot(new_houses, label='Новые дома', color='orange')
plt.title('Сравнение распределения цен на старые и новые дома')
plt.xlabel('Цена')
plt.ylabel('Плотность вероятности')
plt.legend()
plt.grid()
plt.show()

ks_stat, p_value_ks = stats.ks_2samp(old_houses, new_houses)
print(f"Статистика критерия Колмогорова-Смирнова: {ks_stat}")
print(f"p-значение: {p_value_ks}\n")

t_stat, p_value_t = stats.ttest_ind(old_houses, new_houses, equal_var=False)
print(f"t-статистика: {t_stat}")
print(f"p-значение (t-тест): {p_value_t}\n")

f_stat, p_value_f = stats.levene(old_houses, new_houses)
print(f"F-статистика: {f_stat}")
print(f"p-значение (F-тест): {p_value_f}\n")

if p_value_ks < 0.05:
    print("Распределения цен на старые и новые дома различны (по критерию Колмогорова-Смирнова).\n")
else:
    print("Нет оснований отвергать гипотезу о равенстве распределений цен на старые и новые дома.\n")

if p_value_t < 0.05:
    print("Средние цены на старые и новые дома различны (по t-тесту).\n")
else:
    print("Нет оснований отвергать гипотезу о равенстве средних цен на старые и новые дома.\n")

if p_value_f < 0.05:
    print("Дисперсии цен на старые и новые дома различны (по F-тесту).\n")
else:
    print("Нет оснований отвергать гипотезу о равенстве дисперсий цен на старые и новые дома.\n")