import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random


def loadCsv(filePath):
    df= pd.read_csv(filePath,header=None)
    df.rename(columns={0: 'x1', 1: 'x2',2:'t'}, inplace=True)  
    return df
def saveCsv(filepath,data):
     np.savetxt(filepath, data, delimiter=",")

def Normalize(df):
    mean1=df[["x1","x2"]].mean().mean()
    df['x1']=df['x1']-mean1
    df['x2']=df['x2']-mean1
    df['x1']=df['x1']/(df['x1'].std())
    df['x2']=df['x2']/(df['x2'].std())
    return df

def SignNotZero(value):
    if value==0:
         return 1
    else: 
        return np.sign(value)

def validationOfWheigts(w1,w2,theta1,theta2,validation_set):
    pVal=len(validation_set.index)
    sumErrors=0
    for mu in validation_set.index:
        x=np.transpose([[validation_set['x1'][mu], validation_set['x2'][mu]]])
        t=validation_set['t'][mu]
        V=np.tanh(np.matmul(w1,x)-theta1)
        O=np.tanh(np.matmul(w2.T,V)-theta2 )   
        sumErrors=sumErrors+np.abs(SignNotZero(O)-t)

    return sumErrors[0][0]/(2*pVal)

traning_set=loadCsv('/Users/anton_heden/Documents/Programing/FFR135-Artificial-Neural-Networks/HW2/training_set.csv')
validation_set=loadCsv('/Users/anton_heden/Documents/Programing/FFR135-Artificial-Neural-Networks/HW2/validation_set.csv')
validation_set=Normalize(validation_set)
traning_set=Normalize(traning_set)


# Constant
M1=32 
ne = 0.01 
maxIterations=500
classificationErrorBoundary=0.1199

# Initialize
w1=np.random.normal(loc=0.0,scale= 1/np.sqrt(2),size=(M1, 2))
w2=np.random.normal(loc=0,scale=1/np.sqrt(M1),size=(M1,1))


theta1=np.zeros((M1, 1))
theta2=np.zeros((1,1))




for epoch in range(maxIterations):
        traning_set=traning_set.sample(frac=1).reset_index(drop=True)
        for mu in traning_set.index:

            x=np.transpose([[traning_set['x1'][mu], traning_set['x2'][mu]]])
            t=traning_set['t'][mu]

            V=np.tanh(np.matmul(w1,x)-theta1) 
            O=np.tanh(np.matmul(w2.T,V)-theta2)
          
            B=-theta2+np.matmul(w2.T,V)
            Delta=(t-O)*(1-(np.tanh(B))**2)
            dw2=ne*Delta*V
            dTheta2=ne*Delta

            b=-theta1+np.matmul(w1,x)
            gPrimeb=np.asarray(1-(np.tanh(b))**2)
            delta=Delta*w2*gPrimeb
            dw1=ne*delta*x.T
            dTheta1=ne*delta
            
            w1=w1+dw1
            w2=w2+dw2
            theta1=theta1-dTheta1
            theta2=theta2-dTheta2

        C=validationOfWheigts(w1,w2,theta1,theta2,validation_set)
        print("Epoche: ",epoch+1, " With classification error: ",C)
        if C<classificationErrorBoundary:
            print("Found!")
            saveCsv('w1.csv',w1)
            saveCsv('w2.csv',w2)
            saveCsv('t1.csv',theta1)
            saveCsv('t2.csv',theta2)
            break





