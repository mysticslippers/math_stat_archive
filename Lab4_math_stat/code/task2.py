import pandas as pd
import statsmodels.api as sm

data = pd.read_csv("cars93.csv")

X_power = data[['Horsepower']]
y_price = data['Price']

X_power = sm.add_constant(X_power)

model_power = sm.OLS(y_price, X_power).fit()
print(model_power.summary())

X_city = data[['MPG.city']]
y_price = data['Price']

X_city = sm.add_constant(X_city)

model_city = sm.OLS(y_price, X_city).fit()
print(model_city.summary())

x_both = data[['MPG.city', 'MPG.highway']]
y_price = data['Price']

x_both = sm.add_constant(x_both)

model_both = sm.OLS(y_price, x_both).fit()
print(model_both.summary())