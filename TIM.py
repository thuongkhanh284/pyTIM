import numpy as np 
import numpy.matlib as nml
import math

def TrainPGM( X ):
    (n,m) = X.shape
    rank_X = np.linalg.matrix_rank(X)
    if (rank_X < m):
        print(" ERROR : Invalid Input ")
        return -1

    mu = np.mean(X, axis =1)
    X2 = X - nml.repmat(mu,1,m)

    u,s,vh = np.linalg.svd(X2, full_matrices = False)

    u = -1*u

    sm = np.zeros((m,m))
    sm = np.asmatrix(sm)

    for i in range(m):
        sm[i,i] = s[i]


    V = vh.T.conj() 
    print(sm)
    # S(N,:) = []
    sm = np.delete(sm,m-1,axis=1)
    sm = np.delete(sm,m-1,axis=0)
    
    V = np.delete(V,m-1,axis=1)

    
    u = np.delete(u,m-1,axis=1)
    u = -1*u
    
    Q = np.dot(sm , np.transpose(V))



    G = getPathWeightMatrix(m)
    L = getLapacian(G)

    ew,ev = np.linalg.eigh(L,UPLO='L')
    V0 = np.delete(ev,0,axis = 1)

    p1 = np.dot(Q,np.transpose(Q))
    p2 = np.dot(Q,V0)
    
    W = np.dot(np.linalg.inv(p1),p2)
    
    print(V)
    mat = np.zeros((m,1))
    
    # m(j) = Q(:,1)'*W(:,j)/sin(1/N*j*pi+pi*(N-j)/(2*N));
    pi = 3.141592
    for i in range(1,m):
        val1 = np.dot( np.transpose(Q[:,0]), W[:,i-1] )
        val2 = np.sin( (1.0/m)* i * np.pi + np.pi*(m-i)/(2*m))
        #print(val1)
        #print(val2)
        mat[i] = np.asscalar(val1) / val2
    
    model = {'W':W , 'U': u, 'mu': mu, 'num': m, 'mat':mat }
    return model


def getPathWeightMatrix(N):
	G = np.zeros((N,N))
	G = np.asmatrix(G)
	for i in range(N-1):
		G[i,i+1] = 1
		G[i+1,i] = 1
	return G

def getLapacian(G):
	N, M = G.shape
	print(N)
	D= np.zeros((N,N))
	sum_G = np.sum(G,axis = 0)
	for i in range(N):
		D[i,i] = sum_G[0,i]

	L = D - G
	return L

