# from https://github.com/laavanyebahl/3D-Reconstruction-and-Epipolar-Geometry
import numpy as np
from E_and_triangulate import essentialMatrix, triangulate

def camera2(E):
    U,S,V = np.linalg.svd(E)
    m = S[:2].mean()
    E = U.dot(np.array([[m,0,0], [0,m,0], [0,0,0]])).dot(V)
    U,S,V = np.linalg.svd(E)
    W = np.array([[0,-1,0], [1,0,0], [0,0,1]])

    if np.linalg.det(U.dot(W).dot(V))<0:
        W = -W

    M2s = np.zeros([3,4,4])
    M2s[:,:,0] = np.concatenate([U.dot(W).dot(V), U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    M2s[:,:,1] = np.concatenate([U.dot(W).dot(V), -U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    M2s[:,:,2] = np.concatenate([U.dot(W.T).dot(V), U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    M2s[:,:,3] = np.concatenate([U.dot(W.T).dot(V), -U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    
    return M2s
    
def bestM2(pts1, pts2, F, K1, K2):

    E = essentialMatrix(F, K1, K2)

    M1 = np.array([ [ 1,0,0,0 ],
                    [ 0,1,0,0 ],
                    [ 0,0,1,0 ]  ])

    M2_list = helper.camera2(E)

    C1 = K1.dot(M1)

    P_best = np.zeros( (pts1.shape[0],3) )
    M2_best = np.zeros( (3,4) )
    C2_best = np.zeros( (3,4) )
    err_best = np.inf

    error_list = []

    index = 0
    for i in range(M2_list.shape[2]):
        M2 = M2_list[:, :, i]
        C2 = K2.dot(M2)
        P_i, err = triangulate(C1, pts1, C2, pts2)
        error_list.append(err)
        z_list = P_i[:, 2]
        if all( z>0 for z in z_list):
            index = i
            err_best = err
            P_best = P_i
            M2_best = M2
            C2_best = C2
            
    return P_best, C2_best, M2_best, err_best
