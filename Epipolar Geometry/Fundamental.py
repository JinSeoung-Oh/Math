# from https://github.com/laavanyebahl/3D-Reconstruction-and-Epipolar-Geometry

def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation

    pts1_scaled = pts1/M
    pts2_scaled = pts2/M

    A_f = np.zeros((pts1_scaled.shape[0], 9))

    for i in range(pts1_scaled.shape[0]):
        A_f[i, :] = [ pts2_scaled[i,0]*pts1_scaled[i,0] , pts2_scaled[i,0]*pts1_scaled[i,1] , pts2_scaled[i,0], pts2_scaled[i,1]*pts1_scaled[i,0] , pts2_scaled[i,1]*pts1_scaled[i,1] , pts2_scaled[i,1], pts1_scaled[i,0], pts1_scaled[i,1], 1  ]

    # print('A shape: ',A_f.shape)

    u, s, vh = np.linalg.svd(A_f)
    v = vh.T
    f = v[:, -1].reshape(3,3)

    ## NO NEED TO SINGULARIZE, ALREADY BEING SINGULARIZED IN REFINEf
    # f = _singularize(f)
    # print(f)

    # print('rank of f :', np.linalg.matrix_rank(f))

    f = refineF(f, pts1_scaled, pts2_scaled)
    # print('refined f :', f)

    # print('rank of refined f :', np.linalg.matrix_rank(f))

    T =  np.diag([1/M,1/M,1])

    unscaled_F = T.T.dot(f).dot(T)
    # print('unscaled_F :', unscaled_F)

    return unscaled_F


def sevenpoint(pts1, pts2, M):
    # Replace pass by your implementation

    pts1_scaled = pts1 / M
    pts2_scaled = pts2 / M

    A_f = np.zeros((pts1_scaled.shape[0], 9))

    for i in range(pts1_scaled.shape[0]):
        A_f[i, :] = [pts2_scaled[i, 0] * pts1_scaled[i, 0], pts2_scaled[i, 0] * pts1_scaled[i, 1], pts2_scaled[i, 0],
                     pts2_scaled[i, 1] * pts1_scaled[i, 0], pts2_scaled[i, 1] * pts1_scaled[i, 1], pts2_scaled[i, 1],
                     pts1_scaled[i, 0], pts1_scaled[i, 1], 1]

    # print('A: ', A_f)
    # print('A shape: ', A_f.shape)

    u, s, vh = np.linalg.svd(A_f)
    v = vh.T
    f1 = v[:, -1].reshape(3, 3)
    f2 = v[:, -2].reshape(3, 3)

    fun = lambda a: np.linalg.det(a * f1 + (1 - a) * f2)

    a0 = fun(0)
    a1 = (2/3)*( fun(1) - fun(-1))  -  ((fun(2)-fun(-2))/12)
    a2 = 0.5*fun(1) + 0.5*fun(-1) -fun(0)
    a3 = (-1/6)*(fun(1)- fun(-1))  +  (fun(2)-fun(-2))/12

    coeff = [a3, a2, a1, a0]
    # coeff = [a0, a1, a2, a3]   // WRONG
    roots = np.roots(coeff)

    # print('roots: ', roots)

    T = np.diag([1 / M, 1 / M, 1])
    F_list =  np.zeros( (3,3,1) )

    for root in roots:
        if np.isreal(root):
            a = np.real(root)
            F = a*f1 + (1- a)*f2
            # F = refineF(F, pts1_scaled, pts2_scaled)
            unscaled_F = T.T.dot(F).dot(T)
            if np.linalg.matrix_rank(unscaled_F)==3:
                print('---------------------------------------------------------------------------')
                F = refineF(F, pts1_scaled, pts2_scaled)
                unscaled_F = F
            F_list = np.dstack(  (  F_list, unscaled_F)  )

    F_list = F_list[:,:,1:]

    # print('F_list shape: ', F_list.shape)

    return F_list
