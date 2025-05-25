import numpy as np
import pandas as pd
from scipy.stats import t, f
import statsmodels.api as sm
from matplotlib import pyplot as plt

# 1. Загрузка данных
data = pd.read_csv('cars93.csv')
y = data['Price'].values  # Зависимая переменная
X = data[['MPG.city', 'MPG.highway', 'Horsepower']].values  # Независимые переменные

# 2. Добавляем столбец единиц для свободного коэффициента (intercept)
X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])

# 3. Вычисляем коэффициенты по МНК
beta = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y

# Коэффициенты
intercept = beta[0]
coef_mpg_city = beta[1]
coef_mpg_highway = beta[2]
coef_horsepower = beta[3]

print(f"Intercept (свободный коэффициент): {intercept:.4f}")
print(f"Coefficient for MPG.city: {coef_mpg_city:.4f}")
print(f"Coefficient for MPG.highway: {coef_mpg_highway:.4f}")
print(f"Coefficient for Horsepower: {coef_horsepower:.4f}")

# 4. Предположение
y_pred = X_with_intercept @ beta
print("\nПервые 5 предсказанных значений:")
print(y_pred[:5])

y_pred = X_with_intercept @ beta
residuals = y - y_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Предполагаемые значения (Fitted values)")
plt.ylabel("Остатки (Residuals)")
plt.title("Остатки vs Предполагаемые значения")
plt.grid(True)
plt.show()

print("\nПроверим модель с помощью библиотеки.")
X = data[['MPG.city', 'MPG.highway', 'Horsepower']]  # Независимые переменные
y = data['Price']  # Зависимая переменная

# Добавляем константу (аналог столбца единиц в ручной реализации)
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())      # Вывод подробного отчёта
print("\nКоэффициенты (statsmodels):")
print(model.params)

print("\nКоэффициенты (моя реализация):")
print(beta)

print("\nПервое подозрение.")
df = pd.read_csv("cars93.csv")

# Отбор нужных признаков и удаление пропусков
data = df[['MPG.city', 'MPG.highway', 'Horsepower', 'Price']].dropna()

# Формируем X и y
X = data[['MPG.city', 'MPG.highway', 'Horsepower']].values
y = data['Price'].values

# Добавляем свободный коэффициент
X = np.hstack([X, np.ones((X.shape[0], 1))])

# Метод наименьших квадратов вручную
XtX = X.T @ X
Xty = X.T @ y
beta = np.linalg.inv(XtX) @ Xty

# Предсказанные значения и остатки
y_pred = X @ beta
residuals = y - y_pred

# Остаточная сумма квадратов и оценка дисперсии
RSS = np.sum(residuals ** 2)
n, p = X.shape
sigma_squared = RSS / (n - p)

# Ковариационная матрица и стандартная ошибка для horsepower (3-й столбец)
cov_beta = sigma_squared * np.linalg.inv(XtX)
SE_horsepower = np.sqrt(cov_beta[2, 2])

# t-статистика
beta_horsepower = beta[2]
t_stat = beta_horsepower / SE_horsepower

# Критическое значение (t-квантиль) для одностороннего теста при alpha = 0.05
t_critical = t.ppf(0.95, df=n - p)

# Вывод результатов
print(f"Коэффициент при Horsepower: {beta_horsepower:.4f}")
print(f"Стандартная ошибка: {SE_horsepower:.4f}")
print(f"t-статистика: {t_stat:.4f}")
print(f"Критическое значение t (alpha = 0.05): {t_critical:.4f}")

if t_stat > t_critical:
    print("Отвергаем H0: Мощность статистически значимо влияет на цену (положительно).")
else:
    print(" Не отвергаем H0: Нет статистически значимого влияния мощности на цену.")

print("\nВторое подозрение.")
# Формирование X и y
SE_city = np.sqrt(cov_beta[0, 0])  # MPG.city — первый столбец

# Значение коэффициента
beta_city = beta[0]

# t-статистика
t_stat = beta_city / SE_city

# Критическое значение для двустороннего теста при alpha = 0.05
t_crit = t.ppf(1 - 0.025, df=n - p)

# Вывод
print(f"Коэффициент при MPG.city: {beta_city:.4f}")
print(f"Стандартная ошибка: {SE_city:.4f}")
print(f"t-статистика: {t_stat:.4f}")
print(f"Критическое значение t (двусторонний тест, α=0.05): ±{t_crit:.4f}")

if abs(t_stat) > t_crit:
    print("Отвергаем H0: Расход в городе статистически значимо влияет на цену.")
else:
    print(" Не отвергаем H0: Нет статистически значимого влияния расхода в городе на цену.")

print("\nТретье подозрение.")
# Целевая переменная
y = data['Price'].values
n = len(y)

# Полная модель: MPG.city, MPG.highway, Horsepower + Intercept
X_full = data[['MPG.city', 'MPG.highway', 'Horsepower']].values
X_full = np.hstack([X_full, np.ones((n, 1))])  # Добавляем intercept
p_full = X_full.shape[1]
beta_full = np.linalg.inv(X_full.T @ X_full) @ (X_full.T @ y)
resid_full = y - X_full @ beta_full
RSS_full = np.sum(resid_full ** 2)

# Упрощённая модель: только Horsepower + Intercept
X_reduced = data[['Horsepower']].values
X_reduced = np.hstack([X_reduced, np.ones((n, 1))])
p_reduced = X_reduced.shape[1]
beta_reduced = np.linalg.inv(X_reduced.T @ X_reduced) @ (X_reduced.T @ y)
resid_reduced = y - X_reduced @ beta_reduced
RSS_reduced = np.sum(resid_reduced ** 2)

# F-статистика
q = p_full - p_reduced  # число ограничений (обнуляемых коэффициентов)
F_stat = ((RSS_reduced - RSS_full) / q) / (RSS_full / (n - p_full))

# Критическое значение F
alpha = 0.05
F_crit = f.ppf(1 - alpha, dfn=q, dfd=n - p_full)

# Вывод
print(f"RSS полной модели: {RSS_full:.4f}")
print(f"RSS упрощённой модели: {RSS_reduced:.4f}")
print(f"F-статистика: {F_stat:.4f}")
print(f"Критическое значение F (α = 0.05): {F_crit:.4f}")

if F_stat > F_crit:
    print("Отвергаем H₀: Расход в городе и на шоссе влияют на цену.")
else:
    print("Не отвергаем H₀: Нет оснований считать, что расход влияет на цену.")
