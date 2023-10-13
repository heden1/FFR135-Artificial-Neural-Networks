import numpy as np
import pandas as pd

def loadCsv(filePath):
    df= pd.read_csv(filePath,header=None)
    df.rename(columns={0: 'Sepal length', 1: 'Sepal width',2:'Petal length',3:'Petal width'}, inplace=True)  
    return df
def saveCsv(filepath,data):
     np.savetxt(filepath, data, delimiter=",")

def Normalize(df):
    maxValue=df.max().max()
    print(maxValue)
    return df/maxValue


dt=0.02 #seconds
N=3 #number of inputs

