## from https://github.com/laavanyebahl/3D-Reconstruction-and-Epipolar-Geometry

def ransacF(pts1, pts2, M):
    # Replace pass by your implementation

    max_inliers  =  -np.inf
    inliers_best = np.zeros(pts1.shape[0], dtype=bool)
    points_index_best  = None
    threshold = 1e-3

    epochs = 1000
    for e in range(epochs):
        points_index = random.sample(range(0, pts1.shape[0]), 7)
        # print(points_index)
        sevenpoints_1 = []
        sevenpoints_2 = []
        for point in points_index:
            sevenpoints_1.append(pts1[point, :])
            sevenpoints_2.append(pts2[point, :])
        sevenpoints_1 = np.asarray(sevenpoints_1)
        sevenpoints_2 = np.asarray(sevenpoints_2)

        F_list =  sevenpoint(sevenpoints_1, sevenpoints_2, M)
        for j in range(F_list.shape[2]):
            f = F_list[:, :, j]
            num_inliers = 0
            inliers = np.zeros(pts1.shape[0], dtype=bool)
            for k in range(pts1.shape[0]):
                X2 = np.asarray(  [pts2[k,0], pts2[k,1], 1] )
                X1 = np.asarray(  [pts1[k,0], pts1[k,1], 1] )

                if abs(X2.T.dot(f).dot(X1)) < threshold:
                    num_inliers = num_inliers +1
                    inliers[k] = True
                else:
                    inliers[k] = False

            # print(num_inliers)

            if num_inliers>max_inliers:
                max_inliers = num_inliers
                inliers_best = inliers
                points_index_best = points_index

    print('epoch: ', epochs-1, 'max_inliers: ', max_inliers)
    # print('points_index_best: ', points_index_best)

    # RE-DOING EIGHT POINT ALGO AFTER RANSAC WITH INLIER POINTS
    pts1_inliers= pts1[np.where(inliers_best)]
    pts2_inliers= pts2[np.where(inliers_best)]

    F_best_all_inliers = eightpoint(pts1_inliers, pts2_inliers, M)

    return F_best_all_inliers, inliers_best
