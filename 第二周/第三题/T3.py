import cv2
import numpy as np

image = cv2.imread("image.png")
im = cv2.imread("silder.png")
slider = cv2.imread("2.png")
h ,w = im.shape[:2]

slider = cv2.resize(slider, (h,w))

imgAug = image.copy()
ht, wt = slider.shape[:2]

orb = cv2.ORB_create(1000)


kp1, des1 = orb.detectAndCompute(slider, None)
kp2, des2 = orb.detectAndCompute(image, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
matches = bf.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

if len(good) > 4:
    srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    matrix, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5)
    print(matrix)
    if matrix is not None:
        pts = np.float32([[0, 0], [0, ht], [wt, ht], [wt, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)
        print(dst)
        print(w)
        print(int(dst[1][0][0]))
        print(int(dst[1][0][0])+h)
        imgAug[int(dst[1][0][1])-h+5:int(dst[1][0][1]+5), int(dst[1][0][0])-20:int(dst[1][0][0])+w-20] = im
        cv2.imshow("Augmented Image", imgAug)
        cv2.waitKey(0)



