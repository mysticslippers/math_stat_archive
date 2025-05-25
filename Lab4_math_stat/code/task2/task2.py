import pandas as pd
from scipy.stats import f
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_csv('iris.csv')
print(df.head())

df['X'] = df['Sepal.Length'] * df['Sepal.Width'] + df['Petal.Length'] * df['Petal.Width']

# Группы по виду
groups = df.groupby('Species')['X']
grand_mean = df['X'].mean()
k = groups.ngroups
n = len(df)

# SSB (межгрупповая сумма квадратов)
ssb = sum(len(group) * (group.mean() - grand_mean) ** 2 for _, group in groups)

# SSW (внутригрупповая сумма квадратов)
ssw = sum(((group - group.mean()) ** 2).sum() for _, group in groups)

# Степени свободы
df_b = k - 1
df_w = n - k

# Среднеквадратичные
msb = ssb / df_b
msw = ssw / df_w

# F-статистика
f_stat = msb / msw

# Критическое значение на уровне значимости alpha
alpha = 0.05
f_crit = f.ppf(1 - alpha, df_b, df_w)

# Вывод
print(f"F-статистика: {f_stat:.4f}")
print(f"Критическое значение F({df_b}, {df_w}) при alpha = 0.05: {f_crit:.4f}")

if f_stat > f_crit:
    print("Отклоняем H0: средние значимо различаются между группами.")
else:
    print("Не отклоняем H0: статистически значимых различий не обнаружено.")

print("\nПроверим с помощью библиотеки.")
# Построение модели: X зависит от категориального фактора Species
model = smf.ols('X ~ C(Species)', data=df).fit()


anova_table = sm.stats.anova_lm(model, typ=1)

print(anova_table)