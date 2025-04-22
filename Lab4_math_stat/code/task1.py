import numpy as np
import pandas as pd
import statsmodels.api as sm

data = pd.read_csv('cars93.csv')

x = data[['MPG.city', 'MPG.highway', 'Horsepower']]
y = data['Price']

x = sm.add_constant(x)

model = sm.OLS(y, x).fit()

coefficients = model.params

residuals = model.resid
residual_variance = np.var(residuals, ddof=2)

confidence_intervals = model.conf_int(alpha=0.05)

r_squared = model.rsquared

print("Коэффициенты модели:")
print(coefficients)
print("\nОстаточная дисперсия:")
print(residual_variance)
print("\nДоверительные интервалы для коэффициентов:")
print(confidence_intervals)
print("\nКоэффициент детерминации (R²):")
print(r_squared)

hypothesis_test = model.summary()
print(hypothesis_test)