import numpy as np
import random

def SignNotZero(value):
    if value==0:
         return 1
    else: 
        return np.sign(value)

def GenerateCordinates(n):
    X=np.zeros((2**n,n))
    half=2**n
    value=1
    for col in range(n):
        half=half/2
        for row in range(2**n):
            if row%half==0:
                value=1-value
            X[row,col]=value
    return X
def GenerateBooleanFunctionRandom(n):
    rows=2**n
    colums=10**4#2**2**n
    B=np.random.choice([-1, 1],(rows,colums))
    return np.unique(B, axis=1)

def CheckIfCorrect(w,x,theta,t,n):
    for i in range(2**n):
        o=SignNotZero(np.dot(w,x[i,:])-theta)
        if o !=t[i]:
            return 0
    return 1

n=5
X=GenerateCordinates(n)
B=GenerateBooleanFunctionRandom(n)

ne=0.05
numberSepreable=0

for t in B.T:#B.T:
    w=np.random.normal(0,1/n,n)
    theta=0
    for epoch in range(20):
        for i in range (2**n):
            o=SignNotZero(np.dot(w,X[i,:])-theta)

            dTheta=-ne*(t[i]-o)
            dw=ne*(t[i]-o)*X[i,:]
            w=w+dw
            theta=theta+dTheta
    numberSepreable=numberSepreable+CheckIfCorrect(w,X,theta,t,n)

print(numberSepreable)
print(100*(numberSepreable/(len(B.T))),"%")