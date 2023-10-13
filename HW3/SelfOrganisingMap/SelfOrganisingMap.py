import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
def loadCsv(filePath):
    df= pd.read_csv(filePath,header=None)
    df.rename(columns={0: 'Sepal length', 1: 'Sepal width',2:'Petal length',3:'Petal width'}, inplace=True)  
    return df
def saveCsv(filepath,data):
     np.savetxt(filepath, data, delimiter=",")

def Normalize(df):
    maxValue=df.max().max()
    return df/maxValue

def h(j,k,x,y,sigma):
    eucDist=np.linalg.norm([j-x,k-y])
    return np.exp((-1/(2*sigma**2))*eucDist**2 )#np.abs(r(i)-r(io))**2) #where r(i) is the position of neuron i in the output array.

def findWiningNeuronIndex(x,w):
    minDistance=np.inf
    for j in range(40):
        for k in range(40):
            distance =np.linalg.norm(w[j,k,:]- x)
            if distance < minDistance:
                minDistance = distance
                l=j
                m=k
    return l,m

    

    

#load("iris-data.csv")
irisData=loadCsv("/Users/anton_heden/Documents/Programing/FFR135-Artificial-Neural-Networks/HW3/SelfOrganisingMap/iris-data.csv")
irisData=Normalize(irisData)
#print(irisData.head)
#print(irisData.loc[:,['Sepal length']])
## INITILIZE



eta0=0.1
sigma0=10
decayRateEta=0.01
decayRateSigma=0.05

nData=len(irisData)
w=np.random.uniform(0,1,size=(40,40, 4))
for epoch in range(10):
    for p in range(150):
    

        r=np.random.randint(0,nData)
        x=np.array(irisData.iloc[r])
        eta=eta0*np.exp(-decayRateEta*epoch)
        sigma=sigma0*np.exp(-decayRateSigma*epoch)
        #x=x.reshape(1,4,1)
        O=np.matmul(w,x)

        #[x1,y,z] = np.unravel_index(np.argmax(O, axis=None), O.shape)  # returns a tuple 
        [l,m]=findWiningNeuronIndex(x,w)
        #print(x1,y)   
        # Loop through all positions in the matrix
        dw=np.zeros(w.shape)
        
        for j in range(len(O)):
            for k in range(len(O[0])): 
                H=   h(j,k,l,m,sigma)
                diff=(x-w[j,k,:])
                dw[j,k,:]=eta*H*diff
        
        w=w+dw
X=[]
Y=[]
Z=[]
print(w[20,:,1])
for i in range(150):
    x=np.array(irisData.iloc[i])
    #print(x)
    #x=x.reshape(1,4,1)
    O=np.matmul(w,x)
    [l,m]=findWiningNeuronIndex(x,w)
    #[x1,x2,z] = np.unravel_index(np.argmax(O, axis=None), O.shape)  # returns a tuple   
    X.append(l)
    Y.append(m)
    #Z.append(z)

print(X)
print(Y)
#print(Z)
plt.scatter(X,Y)
plt.show()
