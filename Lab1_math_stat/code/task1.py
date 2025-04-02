import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

url = "https://drive.google.com/uc?id=1vv2jGNp6EO8HHRoscDRQU90faR3j8iTN"
data = pd.read_csv(url)

carTypesCount = data['Type'].value_counts()
print("Тип автомобиля и общее количество автомобилей такого типа:\n")
for typeName, count in zip(carTypesCount.index, carTypesCount.values):
    print(f"{typeName}: {count}")
print("")

theMostCommonType = carTypesCount.idxmax()
theLeastCommonType = carTypesCount.idxmin()
print(f"Наиболее распространённый тип автомобиля: {theMostCommonType}")
print(f"Наименее распространённый тип автомобиля: {theLeastCommonType}")


def computeStatistics(group):
    mean = np.mean(group)
    variance = np.var(group, ddof=1)
    median = np.median(group)
    iqr = np.percentile(group, 75) - np.percentile(group, 25)
    return mean, variance, median, iqr


statisticsOverall = computeStatistics(data['MPG.highway'])
statisticsOverall = [round(value, 2) for value in statisticsOverall]
print("\nРасчёт статистических данных мощности автомобиля для всей совокупности:")
print(
    f"Выборочное среднее: {statisticsOverall[0]}, Выборочная дисперсия: {statisticsOverall[1]}, выборочная медиана: {statisticsOverall[2]}, межквартильный размах: {statisticsOverall[3]}")

americanDataSet = data[data['Origin'] == 'USA']
statisticsAmerican = computeStatistics(americanDataSet['MPG.highway'])
statisticsAmerican = [round(value, 2) for value in statisticsAmerican]
print("\nРасчёт статистических данных мощности автомобиля для всех американских автомобилей:")
print(
    f"Выборочное среднее: {statisticsAmerican[0]}, Выборочная дисперсия: {statisticsAmerican[1]}, выборочная медиана: {statisticsAmerican[2]}, межквартильный размах: {statisticsAmerican[3]}")

nonAmericanDataSet = data[data['Origin'] != 'USA']
statisticsNonAmerican = computeStatistics(nonAmericanDataSet['MPG.highway'])
statisticsNonAmerican = [round(value, 2) for value in statisticsNonAmerican]
print("\nРасчёт статистических данных мощности автомобиля для всех не американских автомобилей:")
print(
    f"Выборочное среднее: {statisticsNonAmerican[0]}, Выборочная дисперсия: {statisticsNonAmerican[1]}, выборочная медиана: {statisticsNonAmerican[2]}, межквартильный размах: {statisticsNonAmerican[3]}\n")

plt.figure(figsize=(15, 5))

print("Построение графика эмпирической функции распределения для всей совокупности автомобилей...")
plt.subplot(1, 3, 1)
sns.ecdfplot(data['MPG.highway'])
plt.title('Эмпирическая функция распределения')
plt.xlabel('Мощность')
plt.ylabel('Функция распределения')

print("Построение гистограммы для всей совокупности автомобилей...")
plt.subplot(1, 3, 2)
sns.histplot(data['MPG.highway'], bins=15, kde=True)
plt.title('Гистограмма (вся совокупность)')
plt.xlabel('Мощность')
plt.ylabel('Частота')

print("Построение box-plot'а для всей совокупности автомобилей...")
plt.subplot(1, 3, 3)
sns.boxplot(y=data['MPG.highway'])
plt.title('Box-plot (вся совокупность)')
plt.ylabel('Мощность')

plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 5))

print("Построение графика эмпирической функции распределения для американских/неамериканских...")
plt.subplot(1, 3, 1)
sns.ecdfplot(americanDataSet['MPG.highway'], label='Американские')
sns.ecdfplot(nonAmericanDataSet['MPG.highway'], label='Неамериканские')
plt.title('Эмпирическая функция распределения')
plt.xlabel('Мощность')
plt.ylabel('Функция распределения')
plt.legend()

print("Построение гистограммы для американских/неамериканских...")
plt.subplot(1, 3, 2)
sns.histplot(americanDataSet['MPG.highway'], bins=15, kde=True, color='blue', label='Американские', alpha=0.5)
sns.histplot(nonAmericanDataSet['MPG.highway'], bins=15, kde=True, color='orange', label='Неамериканские', alpha=0.5)
plt.title('Гистограмма (американские/неамериканские)')
plt.xlabel('Мощность')
plt.ylabel('Частота')
plt.legend()

print("Построение box-plot'а для американских/неамериканских...")
plt.subplot(1, 3, 3)
sns.boxplot(x=data['Origin'], y=data['MPG.highway'])
plt.title('Box-plot (американские/неамериканские)')
plt.ylabel('Мощность')

plt.tight_layout()
plt.show()
