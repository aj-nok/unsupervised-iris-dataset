# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 13:20:37 2019

@author: aj-nok
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
model = KMeans(n_clusters=3)

df=pd.read_csv('iris.csv',usecols=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])


df = StandardScaler().fit_transform(df)                  #Feature Scaling
principalComponents = pca.fit_transform(df)              #Dimensionality reduction
df_new = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2'])

model.fit(df_new)
predictions = model.predict(df_new)

x_ax = df_new['pc1']                                     #scatter plot
y_ax = df_new['pc2']  
plt.title('Scatter plot pc1 vs pc2')
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.xticks(())
plt.yticks(())
plt.scatter(x_ax, y_ax, c=predictions)
plt.show()

