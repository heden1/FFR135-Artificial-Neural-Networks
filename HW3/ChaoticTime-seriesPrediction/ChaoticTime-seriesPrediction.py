import numpy as np
import pandas as pd
import sys

def loadCsv(filePath):
    df= pd.read_csv(filePath,header=None)
    return df
def saveCsv(filepath,data):
     np.savetxt(filepath, data, delimiter=",")

def calcOutput(wOut,r):
    return wOut@r

def NextR(wIn,w,r,input):
    term1=w@r
    term2=wIn.T@input
    return np.tanh(term1 + term2)

def RidgeRegression(X,Y,k):
    n=X.shape[1]
    I=np.identity(n)
    return (Y@X)@(np.linalg.inv((X.T@X)+I*k))
  
#Constants 
N=3 #number of inputs
nReservoir=500
k=0.01

#Initilize
wIn=np.random.normal(0,np.sqrt(0.002), size=(N,nReservoir))
w=np.random.normal(0, np.sqrt(2 / nReservoir), size=(nReservoir, nReservoir))
r0=np.ones((nReservoir,1))
traning_set=loadCsv("/Users/anton_heden/Documents/Programing/FFR135-Artificial-Neural-Networks/HW3/ChaoticTime-seriesPrediction/training-set.csv")
traning_set=traning_set.to_numpy()
nInputs=traning_set.shape[1]

test_set=loadCsv("/Users/anton_heden/Documents/Programing/FFR135-Artificial-Neural-Networks/HW3/ChaoticTime-seriesPrediction/test-set.csv")
test_set=test_set.to_numpy()
nTestSet=test_set.shape[1]

R=np.zeros((nReservoir,nInputs-1))
r=r0
for t in range(nInputs-1):
    input_traning=np.array([traning_set[:,t]]).T
    r=NextR(wIn,w,r0,input_traning)
    R[:,t]=np.squeeze(r)

Y=traning_set[:,1:]
wOut=RidgeRegression(R.T,Y,k)

r=r0
rTestList=np.zeros((3,100))
r1List=np.zeros((3,100))
output=calcOutput(wOut,r)
yComponents=[]
runs=500+nTestSet
r=r0
input=np.array([test_set[:,0]]).T
for t in range(runs):
    r=NextR(wIn,w,r0,input)
    output=calcOutput(wOut,r)
    input=output
    yComponents.append(output[1])
saveCsv("prediction.csv",yComponents[100:])