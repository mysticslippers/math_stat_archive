import pandas as pd

data = pd.read_csv('kc_house_data.csv')

correlation = data['sqft_living'].corr(data['price'])

print(f'Коэффициент корреляции: {correlation}')

if correlation > 0.5:
    print("Существует значительная положительная корреляция между площадью и ценой жилья.")
else:
    print("Корреляция между площадью и ценой жилья незначительна.")