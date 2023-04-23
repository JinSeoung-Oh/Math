import cv2
import numpy as np
from matplotlib import pyplot as plt

## get fundamentalMat

sift = cv2.xfeatures2d.SIFT_create()
img1 = cv2.imread('xxx.png')
img2 = cv2.imread('yyy.png')

kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
good = []
pts1 = []
pts2 = []
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)


## view mathing pair from pts1, pts2

imgL = cv2.imread('xxx.png')
imgR = cv2.imread('yyy.png')

(hA, wA) = imgL.shape[:2]
(hB, wB) = imgR.shape[:2]
vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")

vis[0:hA, 0:wA] = imgL
vis[0:hB, wA:] = imgR

for ((x1, y1), (x2, y2)) in zip(pts1, pts2):
    ptA = (int(x1), int(y1))
    ptB = (int(x2) + wA, int(y2))
    cv2.line(vis, ptA, ptB, (0, 255, 0), 3)

plt.imshow(vis)
