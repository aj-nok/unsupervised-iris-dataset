# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 14:30:52 2019

@author: aj-nok
"""


import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv('iris.csv',usecols=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])

x_ax = df['SepalLengthCm']  
y_ax = df['SepalWidthCm']  
plt.title('Scatter plot Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(x_ax, y_ax, c='red')
plt.show()

x_ax = df['PetalLengthCm']  
y_ax = df['PetalWidthCm']  
plt.title('Scatter plot Petal Length vs Petal Width')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.scatter(x_ax, y_ax, c='blue')
plt.show()

x_ax = df['SepalLengthCm']  
y_ax = df['PetalLengthCm']  
plt.title('Scatter plot Sepal Length vs Petal Length')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.scatter(x_ax, y_ax, c='black')
plt.show()

x_ax = df['SepalLengthCm']  
y_ax = df['PetalWidthCm']  
plt.title('Scatter plot Sepal Length vs Petal Width')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Width')
plt.scatter(x_ax, y_ax, c='red')
plt.show()

x_ax = df['PetalLengthCm']  
y_ax = df['SepalWidthCm']  
plt.title('Scatter plot Petal Length vs Sepal Width')
plt.xlabel('Petal Length')
plt.ylabel('Sepal Width')
plt.scatter(x_ax, y_ax, c='blue')
plt.show()

