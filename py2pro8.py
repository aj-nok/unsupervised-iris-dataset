# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:15:38 2019

@author: ajas
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 13:20:37 2019

@author: ajas
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
model = KMeans(n_clusters=3)

df=pd.read_csv('iris.csv',usecols=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])
df_lab=pd.read_csv('iris.csv',usecols=['Species'])
df_lab.loc[df_lab['Species'] == 'Iris-setosa', 'Species'] = 0
df_lab.loc[df_lab['Species'] == 'Iris-versicolor', 'Species'] = 1
df_lab.loc[df_lab['Species'] == 'Iris-virginica', 'Species'] = 2

df = StandardScaler().fit_transform(df)
principalComponents = pca.fit_transform(df)
df_new = pd.DataFrame(data = principalComponents , columns = ['pc1', 'pc2'])
model.fit(df_new)
predictions = model.predict(df_new)
#score = metrics.accuracy_score(df_lab,predictions)
#print('Accuracy:{0:f}'.format(score))

fpr, tpr, thresholds = metrics.roc_curve(df_lab, predictions, pos_label=2)
print(metrics.auc(fpr, tpr))


