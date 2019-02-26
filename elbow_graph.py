# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:28:56 2019

@author: aj-nok
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df=pd.read_csv('iris.csv',usecols=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])
sse = {}
for k in range(1, 15):
    kmeans = KMeans(n_clusters=k).fit(df)
#    print(kmeans.labels_)
    sse[k] = kmeans.inertia_

plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.title("Elbow graph")
plt.show()
