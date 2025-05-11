import pandas as pd
import numpy as np

data = pd.read_csv('exams_dataset.csv')

data['total score'] = data['math score'] + data['reading score'] + data['writing score']

groups = data.groupby('race/ethnicity')['total score'].apply(list)

k = len(groups)

N = data.shape[0]

group_means = [np.mean(group) for group in groups]
global_mean = np.mean(data['total score'])

SSB = N * sum((mean - global_mean) ** 2 for mean in group_means)

SSW = sum(sum((score - mean) ** 2 for score in group) for group, mean in zip(groups, group_means))

df_between = k - 1
df_within = N - k

MSB = SSB / df_between
MSW = SSW / df_within

F = MSB / MSW

from scipy.stats import f

alpha = 0.05

p_value = 1 - f.cdf(F, df_between, df_within)

print(f'F-статистика: {F}')
print(f'p-значение: {p_value}')

if p_value < alpha:
    print("Отвергаем нулевую гипотезу: существует значимая разница между средними значениями в группах.")
else:
    print("Нет оснований для отклонения нулевой гипотезы: средние значения в группах равны.")
