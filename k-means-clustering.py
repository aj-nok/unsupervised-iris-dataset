# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 14:47:52 2019

@author: aj-nok
"""


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd 

df=pd.read_csv('iris.csv',usecols=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])

model = KMeans(n_clusters=2)
model.fit(df)
predictions = model.predict(df)

x_ax = df['SepalLengthCm']  
y_ax = df['SepalWidthCm']  
plt.title('Scatter plot Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(x_ax, y_ax, c=predictions)
plt.show()
