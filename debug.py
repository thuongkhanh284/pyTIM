import numpy as np 
import numpy.matlib as nml
import math
import scipy
import scipy.linalg

X = np.matrix([[1,2],[2,3],[1,4]])

(n,m) = X.shape
rank_X = np.linalg.matrix_rank(X)
if (rank_X < m):
	print(" ERROR : Invalid Input ")


mu = np.mean(X, axis =1)
X2 = X - nml.repmat(mu,1,m)

u,s,vh = np.linalg.svd(X2,full_matrices =False)

u = -1*u

sm = np.zeros((m,m))
sm = np.asmatrix(sm)

for i in range(m):
	sm[i,i] = s[i]


V = vh.T.conj() 
print(u)
print(sm)
print(V)

G = getPathWeightMatrix(m)
L = getLapacian(G)

w,v = np.linalg.eigh(L,UPLO='L')
V0 = np.delete(v,0,axis = 1)