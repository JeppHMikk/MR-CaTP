import math
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from scipy.optimize import linear_sum_assignment

def collisionConstraint(p,r,epsilon,K):
    N = p.shape[1]
    # CALCULATE DELAUNAY GRAPH
    tri = Delaunay(p.transpose())
    D = np.concatenate((tri.simplices[:,[0,1]],tri.simplices[:,[0,2]],tri.simplices[:,[1,2]]),axis=0) # Make triangles into a list of robot pairs
    D = np.sort(D,axis=1) # Sort robot pairs in ascending order
    tmp, indices = np.unique(D,axis=0,return_index=True) # Only keep unique robot pairs
    D = D[indices,:]

    # CALCULATE INEQUALITY CONSTRAINTS
    C = []
    d = []
    for i in range(N):
        D_i = D[np.any(D==i,axis=1),:]
        mask = D_i[:, 1] == i
        D_i[mask] = D_i[mask][:, ::-1]
        n_i = D_i.shape[0]
        C_i = np.zeros(shape=(n_i,2))
        d_i = np.zeros(shape=(n_i,1))
        for k in range(n_i):
            j = D_i[k,1]
            C_i[k,:] = (p[:,j] - p[:,i])/np.linalg.norm(p[:,j] - p[:,i],2)
            d_i[k,:] = 1/2*C_i[k,:].dot(p[:,i] + p[:,j]) - (r + epsilon/2)
        C.append(C_i)
        d.append(d_i)
    C = block_diag(*C)
    d = np.vstack(d)

    L = np.tril(np.ones(shape=(K,K)))
    B = np.kron(L,np.eye(2*N))
    d = np.kron(np.ones(shape=(K,1)),d - C.dot(np.reshape(p,newshape=(2*N,1))))
    C = np.kron(np.eye(K),C).dot(B)

    return C,d

def poiAssignment(p,p_poi,K):
    N = p.shape[1]
    M = p_poi.shape[1]
    C = np.zeros(shape=(N,M))
    for i in range(N):
        for j in range(M):
            C[i,j] = np.linalg.norm(p[:,i] - p_poi[:,j],2)
    row_ind, col_ind = linear_sum_assignment(C)
    S = np.zeros(shape=(N,M))
    S[row_ind,col_ind] = 1
    S = S.transpose()

    return S

def costFun(S,K,p,p_poi,M,eta,zeta):
    n = p.shape[1]
    m = p_poi.shape[1]
    P = np.kron(np.eye(K),np.kron(S,np.eye(2))).dot(np.kron(np.ones(shape=(K,1)),np.reshape(p,newshape=(2*n,1))))
    Gamma = np.kron(np.ones(shape=(K,1)),np.reshape(p_poi,newshape=(2*m,1)))
    E = Gamma - P

    L = np.tril(np.ones(shape=(K,K)))
    B = np.kron(L,np.eye(2*n))

    S_tilde = np.kron(np.eye(K),np.kron(S,np.eye(2))).dot(B)
    H = S_tilde.transpose().dot(S_tilde) + zeta*np.eye(2*n*K)

    r = np.multiply(np.reshape(M[-1,:],newshape=(1,2*n*K)),np.kron(np.ones(shape=(1,K)),1 - np.ones(shape=(1,2*K)).dot(np.kron(S,np.eye(2)))))
    f = E.transpose().dot(S_tilde) + eta*r

    return H


def main():

    N = 10
    M = 5
    K = 5
    p = np.random.uniform(low=-10, high=10, size=(2,N))
    p_poi = np.random.uniform(low=-10, high=10, size=(2,M))
    eta = 1000
    zeta = 0.1

    S = poiAssignment(p,p_poi,K)

    plt.figure()
    plt.plot(p[0,:],p[1,:],'x')

    D = collisionConstraint(p,1,0.1,K)

    M = np.random.uniform(low=-1,high=1,size=(K,2*N*K))

    costFun(S,K,p,p_poi,M,eta,zeta)

    for row in D:
        plt.plot(p[0,row],p[1,row])

    #plt.show()




if __name__ == '__main__':
    main()