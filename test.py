import random
import numpy as np
import pandas as pd
a=np.array([[0, 1, 2]])
a=a.T
print(np.shape(a))
print(np.tanh(a))
print(a)

# initialize list elements
data = [-1,-1]
data2 = [1,2]
  
# Create the pandas DataFrame with column name is provided explicitly
df = pd.DataFrame(data, columns=['x1'])
df['x2']=data2
print(df.mean().mean())
mean1=df[["x1","x2"]].mean().mean()
print(mean1)

np.random.seed(42) 
A = np.random.randint(0, 10, size=(40, 40, 4)) 
B = np.random.randint(0, 10, size=(4)) 
print(B)
B=B.reshape(1,4,1)
print(B)
print(B.shape)
#C=np.matmul(A,B)
#print(C.shape)
cord=np.indices((3,))

print(cord.shape)
a=2
b=2
print(cord*a)

print(np.linalg.norm([[1],[1],[0],[0]]))


#print(np.hypot([0,0],[[0,1],[0,1]]))
#print(np.linalg.norm([[1,1],[0,1]])