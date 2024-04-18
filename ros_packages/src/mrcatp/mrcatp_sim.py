import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial import Delaunay
from scipy.linalg import block_diag
import cvxopt
import sys
import os
from qpsolvers import solve_qp

np.random.seed(17)

def poiAssignment(p,pois):
    N = p.shape[1]
    N_pois = pois.shape[1]
    # CALCULATE DISTANCE BETWEEN EACH ROBOT AND POI FOR HUNGARIAN ALGORITHM
    C = np.zeros(shape=(N-1,N_pois))
    for i in range(N_pois):
        for j in range(1,N):
            C[j-1,i] = np.linalg.norm(pois[:,i] - p[:,j],ord=2)

    # CALCULATE ASSIGNMENT VECTOR AND MATRIX
    row_ind, col_ind = linear_sum_assignment(C)
    S = np.zeros(shape=(N,N_pois))
    S[row_ind+1,col_ind] = 1
    S = S.transpose()
    ass = row_ind+1

    return ass, S

def prr(p1,p2,alpha,d50):
    d = np.linalg.norm(p2-p1,2)
    return 1 - 1/(1 + math.exp(-alpha*(d - d50))) #math.exp(-alpha*(d - d50))/(1 + math.exp(-alpha*(d - d50)))

def adjacencyMatrix(p,alpha,d50):
    N = p.shape[1]
    A = np.zeros(shape=(N,N))
    for i in range(N):
        for j in range(N):
            A[i,j] = prr(p[:,i],p[:,j],alpha,d50)
            A[j,i] = A[i,j]
    return A

def collisionConstraint(p,r,epsilon,K):
    N = p.shape[1]
    tri = Delaunay(p.transpose())
    D = np.concatenate((tri.simplices[:,[0,1]],tri.simplices[:,[0,2]],tri.simplices[:,[1,2]]),axis=0) # Make triangles into a list of robot pairs
    D = np.sort(D,axis=1) # Sort robot pairs in ascending order
    tmp, indices = np.unique(D,axis=0,return_index=True) # Only keep unique robot pairs
    D = D[indices,:]

    for i in range(N):
        D_i = D[np.any(D==i,axis=1),:]
        D_i = D[np.any(D==i,axis=1),:]
        mask = D_i[:, 1] == i
        D_i[mask] = D_i[mask][:, ::-1]
        n_i = D_i.shape[0]
        C_i = np.zeros(shape=(n_i,2))
        d_i = np.zeros(shape=(n_i,1))
        for h in range(n_i):
            j = D_i[h,1]
            C_i[h,:] = (p[:,j] - p[:,i])/np.linalg.norm((p[:,j] - p[:,i]),ord=2)
            d_i[h,:] = .5*C_i[h,:].dot(p[:,j] + p[:,i])
        if(i == 0):
            C = C_i
            d = d_i
        else:
            C = block_diag(C,C_i)
            d = np.concatenate((d,d_i),axis=0)

    # for checking if C is generated correctly
    #p_col = np.reshape(p,newshape=(2*N,1),order='F')
    #print(np.all(C.dot(p_col) <= np.reshape(d,newshape=(d.shape[0],1))))

    d = d - (r + epsilon/2)

    p_col = np.reshape(p,newshape=(2*N,1),order='F')
    L = np.tril(np.ones(shape=(K,K)))
    B = np.kron(L,np.eye(2*N))
    d = np.kron(np.ones(shape=(K,1)),d - C.dot(p_col))
    C = np.kron(np.eye(K),C).dot(B)

    return C,d

def communicationConstraint(p,A,l2,v2,l2_min,alpha,K):
    N = p.shape[1]
    m = np.zeros(shape=(2,N))

    dA = np.zeros(shape=(N,N,2,N))

    for i in range(N):
        for j in range(N):
            if(i != j):
                dij = np.linalg.norm(p[:,i]-p[:,j],ord=2)
                dA[i,j,0,i] = -alpha*(1 - A[i,j])*A[i,j]*(p[0,i] - p[0,j])/dij
                dA[i,j,1,i] = -alpha*(1 - A[i,j])*A[i,j]*(p[1,i] - p[1,j])/dij
                dA[j,i,:,i] = dA[i,j,:,i]
    
    for i in range(N):
        dDi0 = np.diag(np.sum(dA[:,:,0,i],axis=0))
        dDi1 = np.diag(np.sum(dA[:,:,1,i],axis=0))
        dLdi0 = dDi0 - dA[:,:,0,i]
        dLdi1 = dDi1 - dA[:,:,1,i]
        m[0,i] = v2.transpose().dot(dLdi0.dot(v2))
        m[1,i] = v2.transpose().dot(dLdi1.dot(v2))


    #dDdh = np.diag(np.sum(dAdh,axis=0))
    #dLdh = dDdh - dAdh
    #m[0,h] = v2.transpose().dot(dLdh[:,:,0].dot(v2))
    #m[1,h] = v2.transpose().dot(dLdh[:,:,1].dot(v2))
    
    L = np.tril(np.ones(shape=(K,K)))
    M = np.kron(L,np.reshape(m,newshape=(1,2*N),order='F'))
    b = (l2_min - l2)*np.ones(shape=(K,1))
    return m,M,b

def costFun(S,K,p,p_poi,M,eta,zeta):
    n = p.shape[1]
    m = p_poi.shape[1]

    p_col = np.reshape(p,newshape=(2*n,1),order='F')
    p_poi_col = np.reshape(p_poi,newshape=(2*m,1),order='F')

    P = np.kron(np.eye(K),np.kron(S,np.eye(2))).dot(np.kron(np.ones(shape=(K,1)),p_col))
    Gamma = np.kron(np.ones(shape=(K,1)),p_poi_col)
    E = Gamma - P

    L = np.tril(np.ones(shape=(K,K)))
    B = np.kron(L,np.eye(2*n))

    S_tilde = np.kron(np.eye(K),np.kron(S,np.eye(2))).dot(B)
    H = S_tilde.transpose().dot(S_tilde) + zeta*np.eye(2*n*K)

    #f = -E.transpose().dot(S_tilde).transpose()
    r = np.multiply(-np.reshape(np.sum(M,axis=0),newshape=(1,2*n*K)),np.kron(np.ones(shape=(1,K)),1-np.sum(np.kron(S,np.eye(2)),axis=0)))

    f = (-E.transpose().dot(S_tilde) + eta*r).transpose()

    R = np.zeros(shape=(n-1,n))
    R[0:n,0:n-1] = R[0:n,0:n-1] + np.eye(n-1)
    R[0:n,1:n] = R[0:n,1:n] - np.eye(n-1)
    R = np.kron(np.kron(R,np.eye(K)),np.eye(2))
    R = R.transpose().dot(R)
    H = H + 20*R

    #H = np.zeros(shape=(2*n*K,2*n*K))
    #f = np.reshape(-M[-1,:],newshape=(1,2*n*K)).transpose()
    #f = -np.reshape(np.sum(M,axis=0),newshape=(2*n*K,1))

    return H,f

def cvxopt_solve_qp(P, q, G, h, A=None, b=None):
    P = .5 * (P + P.T)  # make sure P is symmetric
    args = [cvxopt.matrix(P), cvxopt.matrix(q)]
    args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
    if A is not None:
        args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    sol = cvxopt.solvers.qp(*args)
    sys.stdout = original_stdout
    if 'optimal' not in sol['status']:
        return None
    return np.array(sol['x']).reshape((P.shape[1],))

def solve_quadratic_program(H, f, C, d):

    # Ensure inputs are numpy arrays
    H = np.asarray(H)
    f = np.asarray(f)
    C = np.asarray(C)
    d = np.asarray(d)

    # Call QuadProg solver
    x = solve_qp(H, f, C, d, solver='quadprog')

    return x

def main():

    N = 10 # number of robots
    tfinal = 200 # simulation duration
    ts = 1 # sampling time of simulator
    dt = 1 # prediction time steps
    T = 200 # number of time steps
    wh = 50 # initial position sampling width and height
    r = 0.1 # robot radii
    epsilon = 10 # minimum clearance
    alpha = 0.1 # signal attenuation
    d50 = 50 # 50% signal attenuation distance
    l2_min = 0.1 # Fiedler value lower bound
    K = 5 # prediction horizon length
    N_pois = 4 # number of inspection points
    reached = False # Boolean for indicating whether all POIs have been reached
    u_max = 1
    eta = 1000
    zeta = 10
    ts = 0.2

    # Generate random initial positions (the first robot is ignored as it is the basestation)
    p = np.zeros(shape=(2,N,T)) # robot positions
    for i in range(1,N):
        while(True):
            pi = np.random.uniform(low=-wh,high=wh,size=(2))
            if(np.all(np.linalg.norm(np.reshape(pi,newshape=(2,1)) - p[:,:,0],ord=2,axis=0) >= r + epsilon/2)):
                p[:,i,0] = pi
                break

    l2 = np.zeros(shape=(T)) # Fiedler value vector

    pois = np.random.uniform(low=-175,high=175,size=(2,N_pois)) # poi positions
    A = np.zeros(shape=(N,N,T)) # adjacency matrix

    # CALCULATE ASSIGNMENT OF ROBOTS TO POIS
    ass, S = poiAssignment(p[:,:,0],pois)

    ax = plt.figure()

    # SIMULATION LOOP
    for k in range(T-1):

        # CHECK IF POIS HAVE BEEN REACHED
        if(np.all(np.linalg.norm(pois - p[:,ass,k],axis=0) <= 1)):
            print("All POIs have been reached")
            reached = True

        # CALCULATE ADJACENCY MATRIX
        A[:,:,k] = adjacencyMatrix(p[:,:,k],alpha,d50) # adjacency matrix
        D = np.diag(np.sum(A[:,:,k],axis=1)) # degree matrix
        L = D - A[:,:,k] # graph laplacian
        
        # CALCULATE CURRENT FIEDLER VALUE AND VECTOR
        l,v = np.linalg.eig(L) # laplacian eigenvalues and eigenvector
        sort_idx = np.argsort(l)
        l = l[sort_idx]
        v = v[:,sort_idx]
        l2[k] = l[1] # Fiedler value
        v2 = np.reshape(v[:,1],newshape=(N,1)) # Fiedler vector

        # CALCULATE COLLISION AVOIDANCE CONSTRAINT
        C_coll,d_coll = collisionConstraint(p[:,:,k],r,epsilon,K)

        # CALCULATE COMMUNICATION CONSTRAINT
        m,C_comm,d_comm = communicationConstraint(p[:,:,k],A[:,:,k],l2[k],v2,l2_min,alpha,K)

        # CALCULATE INPUT CONSTRAINT
        C_in = np.concatenate((-np.eye(2*N*K),np.eye(2*N*K)),axis=0)
        d_in = np.concatenate((u_max*np.ones(shape=(2*N*K,1)),u_max*np.ones(shape=(2*N*K,1))),axis=0)

        # CONCATENATE CONSTRAINTS TOGETHER
        C = np.concatenate((C_coll,-C_comm,C_in),axis=0)
        d = np.concatenate((d_coll,-d_comm,d_in),axis=0)

        # GENERATE COST FUNCTION
        H,f = costFun(S,K,p[:,:,k],pois,C_comm,eta,zeta)

        # SOLVE OPTIMIZATION PROBLEM
        u_opt = solve_quadratic_program(H,f,C,d)
        
        # APPLY FIRST INPUT
        u = np.reshape(u_opt[0:2*N],newshape=(2,N),order='F')

        p_pred = np.zeros(shape=(2,K,N))

        for i in range(K):
            u_opt_i = u_opt[i*N:i*N+2*N]
            u_i = np.reshape(u_opt_i,newshape=(2,N),order='F')
            if(i == 0):
                p_pred[:,i,:] = p[:,:,k] + u_i
            else:
                p_pred[:,i,:] = p_pred[:,i-1,:] + u_i

        #for i in range(N):
        #    for j in range(K):
        #        print(j*2*N+i)
        #        print(j*2*N+i+1)
        #        if(j == 0):
        #            p_pred[:,j,i] = p[:,i,k] + u_opt[j*2*N + i:j*2*N + i + 1]
        #        else:
        #            p_pred[:,j,i] = p_pred[:,j-1,i] + u_opt[j*2*N + i:j*2*N + i + 1]



        # UPDATE ROBOT POSITIONS
        for i in range(N):
            p[:,i,k+1] = p[:,i,k] + u[:,i]

        print(l2[k])

        plt.clf()
        plt.plot(p[0,:,k],p[1,:,k],"r.")
        for i in range(N):
            plt.plot(p[0,i,0:k],p[1,i,0:k],'k')
            plt.plot(p_pred[0,:,i],p_pred[1,:,i],'g')
        plt.plot(pois[0,:],pois[1,:],"rx")
        #plt.axis([-200, 200, -200, 200],"equal")
        plt.axis("equal")
        plt.pause(0.01)

    plt.show()

if __name__ == '__main__':
    main()