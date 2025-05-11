import numpy as np
import pandas as pd
import statsmodels.api as sm

data = pd.read_csv("cars93.csv")
data = data[['MPG.city', 'MPG.highway', 'Horsepower', 'Price']].dropna()

X = data[['MPG.city', 'MPG.highway', 'Horsepower']]
y = data['Price']

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

coefficients = model.params
print("Коэффициенты:")
print(coefficients)

residuals = model.resid
residual_variance = np.var(residuals)
print("\nОстаточная дисперсия:")
print(residual_variance)

conf_int = model.conf_int(alpha=0.05)
print("\nДоверительные интервалы:")
print(conf_int)

r_squared = model.rsquared
print("\nКоэффициент детерминации (R^2):")
print(r_squared)

t_stat_power = model.tvalues['Horsepower']
p_value_power = model.pvalues['Horsepower']

print(f"\nT-статистика мощности: {t_stat_power}, p-значение: {p_value_power}")
if p_value_power < 0.05:
    print("Существует значимая связь между мощностью и ценой автомобиля.")
else:
    print("Нет достаточных оснований утверждать, что мощность влияет на цену автомобиля.")

t_stat_mpg_city = model.tvalues['MPG.city']
p_value_mpg_city = model.pvalues['MPG.city']

print(f"\nT-статистика MPG.city: {t_stat_mpg_city}, p-значение: {p_value_mpg_city}")
if p_value_mpg_city < 0.05:
    print("Существует значимая связь между расходом в городе и ценой автомобиля.")
else:
    print("Нет достаточных оснований утверждать, что расход в городе влияет на цену автомобиля.")

f_test = model.fvalue
f_p_value = model.f_pvalue

print(f"\nF-статистика: {f_test}, p-значение для F-теста: {f_p_value}")
if f_p_value < 0.05:
    print("Модель значима, по крайней мере один из факторов (мощность или расход) влияет на цену.")
else:
    print("Модель не объясняет значительное количество вариации в цене автомобиля.")
