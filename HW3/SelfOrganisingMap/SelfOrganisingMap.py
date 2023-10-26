import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
def loadCsv(filePath):
    df= pd.read_csv(filePath,header=None)
    return df

def Normalize(df):
    maxValue=df.max().max()
    return df/maxValue

def h(j,k,x,y,sigma):
    eucDist=np.linalg.norm([j-x,k-y])
    return np.exp((-1/(2*sigma**2))*eucDist**2 )

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

## Load data
irisData=loadCsv("/Users/anton_heden/Documents/Programing/FFR135-Artificial-Neural-Networks/HW3/SelfOrganisingMap/iris-data.csv")
irisData=Normalize(irisData)
irisLabel=loadCsv("/Users/anton_heden/Documents/Programing/FFR135-Artificial-Neural-Networks/HW3/SelfOrganisingMap/iris-labels.csv")

## INITILIZE
eta0=0.1
sigma0=10
decayRateEta=0.01
decayRateSigma=0.05
nData=len(irisData)
w=np.random.uniform(0,1,size=(40,40, 4))

###############
# Finding closest neron (untrained)
###############
X1=[]
Y1=[]
L1=[]
for i in range(150):
    x=np.array(irisData.iloc[i])
    label1=np.array(irisLabel.iloc[i])
    [l,m]=findWiningNeuronIndex(x,w)
    X1.append(l)
    Y1.append(m)
    L1.append(label1)

###############
# Ploting
###############
figure, axis = plt.subplots(1, 2)
scatter0=axis[0].scatter(X1,Y1,s=50,c=L1)
axis[0].set_title("Random") 
legend0 = axis[0].legend(*scatter0.legend_elements(), title="Classes",bbox_to_anchor=(1.13, 1.0))
axis[0].add_artist(legend0)

########################
#Training
########################
for epoch in range(10):
    irisData=irisData.sample(frac=1).reset_index(drop=True) #Stochastic (shuffle)
    for p in range(150):
        x=np.array(irisData.iloc[p])
        eta=eta0*np.exp(-decayRateEta*epoch)
        sigma=sigma0*np.exp(-decayRateSigma*epoch)
        [x1,y]=findWiningNeuronIndex(x,w)
        dw=np.zeros(w.shape)
        
        for j in range(40):
            for k in range(40): 
                H=h(j,k,x1,y,sigma)
                diff=(x-w[j,k,:])
                dw[j,k,:]=eta*H*diff
        w=w+dw

###############
# Finding closest neron (trained)
###############
X=[]
Y=[]
L=[]
irisData=loadCsv("/Users/anton_heden/Documents/Programing/FFR135-Artificial-Neural-Networks/HW3/SelfOrganisingMap/iris-data.csv")
irisData=Normalize(irisData)
for i in range(150):
    x=np.array(irisData.iloc[i])
    label=np.array(irisLabel.iloc[i])
    [l,m]=findWiningNeuronIndex(x,w)
    X.append(l)
    Y.append(m)
    L.append(label)

###############
# Ploting result
###############
scatter=axis[1].scatter(X,Y,s=50,c=L)
axis[1].set_title("Trained") 
legend1 = axis[1].legend(*scatter.legend_elements(), title="Classes",bbox_to_anchor=(1.13, 1))
axis[1].add_artist(legend1)
plt.show()
