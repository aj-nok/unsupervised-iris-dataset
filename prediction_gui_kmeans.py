# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:02:27 2019

@author: aj-nok
"""

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import *
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class mclass:
    def __init__(self,  window):
         self.window = window      
         self.oper = Label(window,text="PLEASE ENTER THE FIELDS:").grid(row=0,column=0,columnspan=2,sticky=W)
         self.data1 = StringVar()
         self.oper = Label(window,text="Sepal Length Cm").grid(row=2,column=0,sticky=W)
         self.insertEntry = Entry(window, textvariable=self.data1).grid(row=2,column=1,sticky=W)
         self.data2 = StringVar()
         self.oper = Label(window,text="Sepal Width Cm").grid(row=4,column=0,sticky=W)
         self.insertEntry = Entry(window, textvariable=self.data2).grid(row=4,column=1,sticky=W)
         self.data3 = StringVar()
         self.oper = Label(window,text="Petal Length Cm").grid(row=6,column=0,sticky=W)
         self.insertEntry = Entry(window, textvariable=self.data3).grid(row=6,column=1,sticky=W)
         self.data4 = StringVar()
         self.oper = Label(window,text="Petal Width Cm").grid(row=8,column=0,sticky=W)
         self.insertEntry = Entry(window, textvariable=self.data4).grid(row=8,column=1,sticky=W)
         self.DisplayButton = Button(window,text = "CHECK",bg="orange", fg="red",command = self.plot).grid(row=4,column=3,columnspan=2)    

    def plot (self):
        seplen = float(self.data1.get())
        sepwid = float(self.data2.get())
        petlen = float(self.data3.get())
        petwid = float(self.data4.get())  
        d= np.array([[seplen,sepwid,petlen,petwid]])
        columns_new = ['SepalLengthCm', 'SepalWidthCm','PetalLengthCm','PetalWidthCm']
        ndf=pd.DataFrame(d, columns=columns_new) 
#        print(ndf)
      
        pca = PCA(n_components=2)
        model = KMeans(n_clusters=3)
        df=pd.read_csv('iris.csv',usecols=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])
        df=df.append(ndf,ignore_index=True)
#        print(df)   
        df = StandardScaler().fit_transform(df)
        principalComponents = pca.fit_transform(df)
#        print(principalComponents)
        df_new = pd.DataFrame(data = principalComponents , columns = ['pc1', 'pc2'])
#        print(df_new)
        model.fit(df_new)
        predictions = model.predict(df_new)
        predictions=predictions[:150]
#        print(predictions[:150])
               
        fig = Figure(figsize=(6,6))                         #Plotting graph in GUI
        a = fig.add_subplot(111)
        x_ax = df_new['pc1'].head(150)  
        y_ax = df_new['pc2'].head(150)
        x_n=df_new['pc1'].tail(1)
        y_n=df_new['pc2'].tail(1)
        a.scatter(x_ax, y_ax, c=predictions)
        a.scatter(x_n, y_n, c='red')
        canvas = FigureCanvasTkAgg(fig, master=self.window)
        canvas.get_tk_widget().grid(row=9,column=24)
        canvas.draw()

window= Tk()
start= mclass (window)
window.mainloop()
