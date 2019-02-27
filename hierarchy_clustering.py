# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 15:31:38 2019

@author: aj-nok
"""

import numpy as np
import pandas as pd 
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

df=pd.read_csv('iris.csv',usecols=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])
label = np.array(pd.read_csv('iris.csv',usecols=['Species']))
#print(df.head())
#print(label)

X = linkage(df, 'ward')                         
#X = linkage(df, 'single')
#X = linkage(df, 'complete')
#X = linkage(df, 'average')
#X = linkage(df, 'centroid')

plt.figure(figsize=(25, 10))
plt.title(' Hierarchical Clustering Dendrogram ')
plt.xlabel('Iris Data')
plt.ylabel('distance')
dendrogram(
    X,
    labels=label,
    leaf_rotation=90.,
    leaf_font_size=8.,
)
plt.show()
