import numpy as np
A=np.array([[1,2,3,4,5],[-1,3,2,4,4]])
B=np.array([[1,3,4,1,2],[-3,2,3,6,1]])
print(np.matmul(A.T,B))
print(A.T@B)