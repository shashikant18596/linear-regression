import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import linear_model
df = pd.read_csv('E:\Data_Science\linear.csv')
print(df)
x = df['area']
y = df['price']
plt.xlabel('Area')
plt.ylabel('Price')
plt.title("Price A/c to Area")
plt.axis([2500,4200,500000,750000])
plt.plot(x,y,"ro-",linewidth = 3,label = 'price of Land')
plt.legend(loc= 0)
plt.show()
model = linear_model.LinearRegression()
model.fit(df[['area']],y)
predicted = model.predict([[3300]])
print(predicted)
print(model.coef_)
print(model.intercept_)
df2 = pd.read_csv("E:\Data_Science\Area.csv")
print(df2)
predicted_value2 = model.predict(df2)
print(predicted_value2)
df2['price'] = predicted_value2
print(df2)
plt.xlabel('Area')
plt.ylabel('Price')
plt.title("Price A/c to Area")
plt.axis([1000,6000,300000,1000000])
plt.plot(df2['areas'],df2['price'],"bo--")
plt.show()