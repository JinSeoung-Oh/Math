def essentialMatrix(F, K1, K2):
    # from equation
    E = K2.T.dot(F).dot(K1)
    return E
  

  
  def triangulate(camera_max1, points_1, Camera_max2, ponit_2):
    #http://cmp.felk.cvut.cz/cmp/courses/TDV/2012W/lectures/tdv-2012-07-anot.pdf
    P = []

    for i in range(points_1.shape[0]):
        A = np.array([   points_1[i,0]*camera_max1[2,:] - camera_max1[0,:] ,
                         points_1[i,1]*camera_max1[2,:] - camera_max1[1,:] ,
                         points_2[i,0]*camera_max2[2,:] - camera_max2[0,:] ,
                         points_2[i,1]*camera_max2[2,:] - camera_max2[1,:]   ])

        u, s, vh = np.linalg.svd(A)
        v = vh.T
        X = v[:,-1]
        X = X/X[-1]
        P.append(X)

    P = np.asarray(P)
    point1_out = np.matmul(camera_max1, P.T )
    point2_out = np.matmul(camera_max2, P.T )

    point1_out = point1_out.T
    point2_out = point2_out.T

    # NORMALIZING
    for i in range(point1_out.shape[0]):
        point1_out[i,:] = point1_out[i,:] / point1_out[i, -1]
        point2_out[i,:] = point2_out[i,:] / point2_out[i, -1]

    # NON - HOMOGENIZING
    point1_out = point1_out[:, :-1]
    point2_out = point2_out[:, :-1]

    # NON-HOMOGENIZING
    P = P[:, :-1]

    return P
