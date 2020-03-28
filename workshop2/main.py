import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  

path_data_csv = 'data/data.csv'

data_x = pd.read_csv(path_data_csv, usecols=['Meses']).to_numpy()
data_y = pd.read_csv(path_data_csv, usecols=['Unidades  de producción en una industria']).to_numpy() 

x_avg = np.mean(data_x, 0)[0]
y_avg = np.mean(data_y, 0)[0]

n = 12
x_squared_sum = np.sum(np.square(data_x))
y_squared_sum = np.sum(np.square(data_y))

x_sum = np.sum(data_y)
y_sum = np.sum(data_y)

x_y_product_sum = sum(data_x[i] * data_y[i] for i in range(n))[0]

b = (x_y_product_sum - n * x_avg * y_avg)/(x_squared_sum - n * x_avg**2)
a = y_avg - b * x_avg

y_13 = b * 13 + a

standard_error_estimate = ((y_squared_sum-a*y_sum-b*x_y_product_sum)/(n-2))**0.5
coef_corr = sum((data_x[i] - x_avg) * (data_y[i] - y_avg) for i in range(n))/(sum((data_x[i] - x_avg)**2 for i in range(n)) * sum((data_y[i] - y_avg)**2 for i in range(n)))**0.5 

form = lambda x: b * x + a
y = form(data_x)
plt.plot(data_x, data_y, 'o-',c='#ff7f0e', label='Demanda')
plt.plot(data_x, y,  'o-', label='Pronóstico')
plt.legend(loc="upper left")
plt.xlabel('Periodo')
plt.ylabel('Unidades de producción')
plt.show()  
print('Error estándar del estimado', standard_error_estimate)
print('Coeficiente de correlación', coef_corr)