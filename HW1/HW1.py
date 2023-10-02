import numpy as np
import random


k=0 

n=4
rows=2**n
B=np.zeros((2**n,2**2**n))

val=1
half=2**2**n
while k < 2**n:
    half=half/2
    m=0
    while m < 2**2**n:
        if m%half==0:
            val=1-val
        B[k,m]=val
        m=m+1
    k=k+1
        
        
        
        
print(np.shape(B))

# 2d exampel https://towardsdatascience.com/perceptron-learning-algorithm-d5db0deab975
x1=np.array([0,1,0,1])
x2= np.array([0,0,1,1])
ore=np.array([0,1,1,1])

input=np.array[[0,0,0],[1,0,1],[0,1,1],[1,1,1]]


ne=0.05
w=np.random.normal(0,1/n,2)
print(w)



for i in range(len(input)):

    for round in range(20):
        i=np.random.randint(4)
        x=input[i]
        if x[2]==1 and np.dot(w,x)-theta<0:
            w=w+n*x
        elif x[2]==0 and np.dot(w,x1)-theta>=0:
            w=w-n*x
        
        


            
        




#while length(B)<2**2**n: #2**2**n
#for k in range(4):
 #   for m in range(2**2**n):
  #      b


 

#print (random.getrandbits(1))

#O=np.sign(np.dot(w*x)-theta)

#whiegts assigned by hebbs rule source: course book p 74

#np.sign(b)
          
