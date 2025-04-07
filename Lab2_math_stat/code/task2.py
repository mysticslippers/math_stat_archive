import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Настройки для графиков
sns.set(style="whitegrid")

theta_val = 5  # Параметр θ
true_theta_sq = theta_val**2  # Истинное значение θ^2
alpha_val = 0.05  # Уровень значимости
num_experiments = 1000  # Количество экспериментов

def simulate_experiment(sample_size):
    confidence_level = 1 - alpha_val
    z_critical_value = stats.norm.ppf((1 + confidence_level) / 2)  # Квантиль нормального распределения

    # Генерация случайных выборок из равномерного распределения
    data_samples = np.random.uniform(low=-theta_val, high=theta_val, size=(num_experiments, sample_size))
    means = np.mean(data_samples, axis=1, keepdims=True)
    variances = np.var(data_samples, axis=1, ddof=1)  # Выборочная дисперсия
    theta_squared_estimates = 3 * variances  # Оценка θ^2

    # Четвертый центральный момент
    central_moment_fourth = np.mean((data_samples - means)**4, axis=1)
    estimated_sigma_squared = central_moment_fourth - variances**2

    # Проверка на отрицательные значения sigma_squared
    estimated_sigma_squared = np.maximum(estimated_sigma_squared, 0)

    # Вычисление длины интервала
    margin_error = z_critical_value * np.sqrt(9 * estimated_sigma_squared) / np.sqrt(sample_size)

    # Левые и правые границы доверительного интервала
    left_bound = theta_squared_estimates - margin_error
    right_bound = theta_squared_estimates + margin_error

    # Подсчет покрытия истинного параметра в доверительном интервале
    correct_intervals = np.sum((true_theta_sq >= left_bound) & (true_theta_sq <= right_bound))
    coverage_probability = correct_intervals / num_experiments
    interval_sizes = 2 * margin_error  # Длина интервала

    return {
        'sample_size': sample_size,
        'coverage_probability': coverage_probability,
        'interval_sizes': interval_sizes
    }

# Проведение эксперимента с разными размерами выборки
experiment_10 = simulate_experiment(10)
experiment_100000 = simulate_experiment(100000)

# Вывод результатов
print(f"Размер выборки 10, покрытие: {experiment_10['coverage_probability']*100:.2f}%")
print(f"Размер выборки 100000, покрытие: {experiment_100000['coverage_probability']*100:.2f}%")

# Визуализация результатов с помощью boxplot
plt.figure(figsize=(10, 5))
sns.boxplot(data=pd.DataFrame({
    'n = 10': experiment_10['interval_sizes'],
    'n = 100000': experiment_100000['interval_sizes']
}))
plt.title("Сравнение длин доверительных интервалов для оценки θ^2")
plt.ylabel("Длина доверительного интервала")
plt.grid(True)
plt.tight_layout()
plt.show()
