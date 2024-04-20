#!/usr/bin/env python
# license removed for brevity
import rospy
import math
import numpy as np
from scipy.spatial import Delaunay
from scipy.linalg import block_diag
from scipy.optimize import linear_sum_assignment
import cvxopt
import sys
import os
from qpsolvers import solve_qp

from std_msgs.msg import String
from nav_msgs.msg import Odometry
from mrs_msgs.msg import TrajectoryReference, Reference, ReferenceStamped, VelocityReferenceStamped
from geometry_msgs.msg import _Pose
from geometry_msgs.msg import PoseWithCovariance
from geometry_msgs.msg import Point
from std_srvs.srv import Trigger, TriggerResponse
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

np.random.seed(17)

activate = False

def pos_callback(p_slice,data):
    #rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)   
    p_slice[:] = np.reshape([data.pose.pose.position.x, data.pose.pose.position.y],newshape=p_slice[:].shape)

def actsrv_callback(req):
    global activate
    activate = True
    rospy.loginfo("mrcatp node activated")
    return TriggerResponse(
        success=True,
        message="mrcatp node activated"
    )

def poiAssignment(p,pois):
    N = p.shape[1]
    N_pois = pois.shape[1]
    # CALCULATE DISTANCE BETWEEN EACH ROBOT AND POI FOR HUNGARIAN ALGORITHM
    C = np.zeros(shape=(N,N_pois))
    for i in range(N_pois):
        for j in range(N):
            C[j,i] = np.linalg.norm(pois[:,i] - p[:,j],ord=2)

    # CALCULATE ASSIGNMENT VECTOR AND MATRIX
    row_ind, col_ind = linear_sum_assignment(C)
    S = np.zeros(shape=(N,N_pois))
    S[row_ind,col_ind] = 1
    S = S.transpose()
    ass = row_ind

    return ass, S

def prr(p1,p2,alpha,d50):
    d = np.linalg.norm(p2-p1,2)
    return 1 - 1/(1 + math.exp(-alpha*(d - d50)))

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

    r = np.multiply(-np.reshape(np.sum(M,axis=0),newshape=(1,2*n*K)),np.kron(np.ones(shape=(1,K)),1-np.sum(np.kron(S,np.eye(2)),axis=0)))
    f = (-E.transpose().dot(S_tilde) + eta*r).transpose()

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

def solve_quadratic_program(H, f, C, d, x0):

    # Ensure inputs are numpy arrays
    H = np.asarray(H)
    f = np.asarray(f)
    C = np.asarray(C)
    d = np.asarray(d)

    # Call QuadProg solver
    x = solve_qp(H, f, C, d, solver='quadprog', initvals=x0)

    return x

def mrcatp():

    rospy.init_node('mrcatp', anonymous=True)

    actsrv = rospy.Service('/mrcatp_activate', Trigger, actsrv_callback)

    N = rospy.get_param("/N") # Number of robots
    N_pois = rospy.get_param("/N_pois")
    uavs = rospy.get_param("/uavs") # Robot names
    alpha = rospy.get_param("/alpha") # PRR attenuation
    d50 = rospy.get_param("/d50") # PRR 50% distance
    K = rospy.get_param("/K") # Prediction horizon length
    r = rospy.get_param("/r") # Robot radius
    epsilon = rospy.get_param("/epsilon") # Collision avoidance clearance
    l2_min = rospy.get_param("/l2_min") # Fiedler value lower bound
    eta = rospy.get_param("/eta")
    zeta = rospy.get_param("/zeta")
    u_max = rospy.get_param("/u_max")
    pois_x = np.asarray(rospy.get_param("/pois_x")).reshape((1,N_pois))
    pois_y = np.asarray(rospy.get_param("/pois_y")).reshape((1,N_pois))
    control_frame = rospy.get_param("/control_frame")
    global activate

    p = np.zeros(shape=(2,N)) # position vectors

    pois = np.concatenate((pois_x,pois_y),axis=0)

    rate = rospy.Rate(20) # 10hz
    
    # INITIALISE ROBOT POSITION ESTIMATE SUBSCRIBERS
    subscribers = []
    for i in range(N):
        p_slice = p[:, i]
        callback_i = lambda data, slice=p_slice: pos_callback(slice, data)
        subscribers.append(rospy.Subscriber("/%s/estimation_manager/odom_main" % uavs[i], Odometry, callback_i))

    publishers = []
    for i in range(N):
        publishers.append(rospy.Publisher("/%s/control_manager/reference" % uavs[i], ReferenceStamped, queue_size=1))

    marker_pub = []
    for id in range(N_pois):
        marker_pub.append(rospy.Publisher('poi_'+str(id), Marker, queue_size=1))

    marker = Marker()
    marker.header.frame_id = "simulator_origin"  # Assuming your marker is in the "map" frame
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.scale.x = 0.5  # Marker size
    marker.scale.y = 0.5
    marker.scale.z = 0.5
    marker.color.a = 0.5
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0

    iter = 0

    u_opt = np.zeros(shape=(2*N*K,1))

    # MAIN LOOP
    while (not rospy.is_shutdown()):

        for i in range(N_pois):
            marker.pose.position.x = pois[0,i]
            marker.pose.position.y = pois[1,i]
            marker.pose.position.z = 1.5
            marker_pub[i].publish(marker)

        if(activate):

            if(iter == 0):
                p_ref = p
                ass, S = poiAssignment(p,pois)

            p_col = np.reshape(p,newshape=(2*N,1),order='F')
            p_poi_col = np.reshape(pois,newshape=(2*N_pois,1),order='F')
            p_insp = np.kron(S,np.eye(2)).dot(p_col)
            e_poi = np.max(np.abs(p_poi_col - p_insp))
            if(e_poi <= 0.1):
                activate = False

            e = np.linalg.norm(p[0:2,:] - p_ref[0:2,:],ord=2,axis=0)

            if(np.all(e <= 0.2)):

                p_curr = p

                # CALCULATE ADJACENCY MATRIX
                A = adjacencyMatrix(p_curr,alpha,d50) # adjacency matrix
                D = np.diag(np.sum(A,axis=1)) # degree matrix
                L = D - A # graph laplacian
                
                # CALCULATE CURRENT FIEDLER VALUE AND VECTOR
                l,v = np.linalg.eig(L) # laplacian eigenvalues and eigenvector
                sort_idx = np.argsort(l)
                l = l[sort_idx]
                v = v[:,sort_idx]
                l2 = l[1] # Fiedler value
                v2 = np.reshape(v[:,1],newshape=(N,1)) # Fiedler vector

                #print(l2)

                # CALCULATE COLLISION AVOIDANCE CONSTRAINT
                C_coll,d_coll = collisionConstraint(p_curr,r,epsilon,K)

                # CALCULATE COMMUNICATION CONSTRAINT
                m,C_comm,d_comm = communicationConstraint(p_curr,A,l2,v2,l2_min,alpha,K)

                # CALCULATE INPUT CONSTRAINT
                C_in = np.concatenate((-np.eye(2*N*K),np.eye(2*N*K)),axis=0)
                d_in = np.concatenate((u_max*np.ones(shape=(2*N*K,1)),u_max*np.ones(shape=(2*N*K,1))),axis=0)

                # CONCATENATE CONSTRAINTS TOGETHER
                C = np.concatenate((C_coll,-C_comm,C_in),axis=0)
                d = np.concatenate((d_coll,-d_comm,d_in),axis=0)

                # GENERATE COST FUNCTION
                H,f = costFun(S,K,p_curr,pois,C_comm,eta,zeta)

                # SOLVE OPTIMIZATION PROBLEM
                u_opt = solve_quadratic_program(H,f,C,d,None)
                
                if(u_opt is None):
                    u_opt = np.zeros(shape=(2*N*K,1))

                # APPLY FIRST INPUT
                u = np.reshape(u_opt[0:2*N],newshape=(2,N),order='F')
                        
                p_ref = p_curr + u

            for i in range(N):
                p_ref_i = ReferenceStamped()
                p_ref_i.reference.position.x = p_ref[0,i]
                p_ref_i.reference.position.y = p_ref[1,i]
                p_ref_i.reference.position.z = 1.5
                p_ref_i.header.frame_id = control_frame
                publishers[i].publish(p_ref_i)

            rate.sleep()

            iter = iter + 1

if __name__ == '__main__':
    try:
        mrcatp()
    except rospy.ROSInterruptException:
        pass