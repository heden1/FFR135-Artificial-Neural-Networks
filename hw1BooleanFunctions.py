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


def GenrateBooleanFunctions(n):
    rows=2**n
    B=np.zeros((2**n,2**2**n))
    k=0 
    val=1
    half=2**2**n
    while k < 2**n:
        half=half/2
        m=0
        while m < 2**2**n:
            if m%half==0:
                val=-val
            B[k,m]=val
            m=m+1
        k=k+1
    print(np.shape(B))
    return (B)  

def CheckIfCorrect(w,x,theta,t,n):
    for i in range(2**n):
        o=SignNotZero(np.dot(w,x[i,:])-theta)#.astype(int)
        if o !=t[i]:
            #print("not sepreabl")
            return 0
    #print("sepreqbel")
    return 1

n=4
X=GenerateCordinates(n)
B=GenrateBooleanFunctions(n)
#print(B)
u=1
#print(B[:,u])
sepreqbelT=np.array([[]])


#x=np.array([[0,0],[0,1],[0,1],[1,1]])
ne=0.05
numberSepreable=0
#print(X[1,:])
for t in B.T:

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
while False:
    for i in range(length(w)):
        dw=ne*(t-o)*x
        w[i]=w[i]-dw
    do = -ne*(t-o) 
    o=o-do 
    


""" hen
for i in range(len(input)):

    for round in range(20):
        i=np.random.randint(4)
        x=input[i]
        if x[2]==1 and np.dot(w,x)-theta<0:
            w=w+n*x
        elif x[2]==0 and np.dot(w,x1)-theta>=0:
            w=w-n*x
        
 """
   


            
        




#while length(B)<2**2**n: #2**2**n
#for k in range(4):
 #   for m in range(2**2**n):
  #      b


 

#print (random.getrandbits(1))

#O=np.sign(np.dot(w*x)-theta)

#whiegts assigned by hebbs rule source: course book p 74

#np.sign(b)
          
