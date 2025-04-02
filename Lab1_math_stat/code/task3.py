import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Фиксируем параметры нормального распределения
theta_0 = 2
sigma = 1  # стандартное отклонение

# Задаём массив объемов выборки
sample_sizes = [10, 50, 100, 500, 1000]
M = 1000  # Количество выборок для каждого n

# Функция для генерации выборок и оценки theta_hat
def estimate_theta(sample_size, M, theta_0, sigma):
    estimates = []
    for _ in range(M):
        sample = np.random.normal(theta_0, sigma, sample_size)
        theta_hat = np.mean(sample)
        estimates.append(theta_hat)
    return np.array(estimates)

# Хранение оценок theta_hat для разных n
estimations = {n: estimate_theta(n, M, theta_0, sigma) for n in sample_sizes}

# Визуализация результатов
fig, axes = plt.subplots(len(sample_sizes), 3, figsize=(15, 4 * len(sample_sizes)))

for i, n in enumerate(sample_sizes):
    data = estimations[n]

    # Гистограмма
    sns.histplot(data, bins=30, kde=True, ax=axes[i, 0])
    axes[i, 0].set_title(f'Гистограмма оценок, n={n}')

    # Box-plot
    sns.boxplot(x=data, ax=axes[i, 1])
    axes[i, 1].set_title(f'Box-plot, n={n}')

    # Violin-plot
    sns.violinplot(x=data, ax=axes[i, 2])
    axes[i, 2].set_title(f'Violin-plot, n={n}')

plt.tight_layout()
plt.show()

# Вывод описательных статистик
stats_df = pd.DataFrame({n: estimations[n] for n in sample_sizes})
stats_summary = stats_df.describe()
stats_summary.index = [
    'Количество выборок',
    'Среднее значение',
    'Стандартное отклонение',
    'Минимальное значение',
    'Первый квартиль (Q1)',
    'Медиана (Q2)',
    'Третий квартиль (Q3)',
    'Максимальное значение'
]

print(stats_summary.T)
