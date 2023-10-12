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