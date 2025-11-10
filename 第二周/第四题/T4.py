import cv2
import numpy as np

def on_trackbar(val):
    pass
def spliter():
    thresh1 = cv2.getTrackbarPos("Threshold1", "Trackbars")
    thresh2 = cv2.getTrackbarPos("Threshold2", "Trackbars")
    clip_w = cv2.getTrackbarPos("Clip_w", "Trackbars")
    clip_h = cv2.getTrackbarPos("Clip_h", "Trackbars")
    c = cv2.getTrackbarPos("save", "Trackbars")
    return thresh1, thresh2,clip_w, clip_h,c

def bigger(contours_):
    biggest = np.array([])
    max_area = 0
    area_1 = 0
    for i in contours_:
        area_2 = cv2.contourArea(i)
        if area_2 > area_1:
            peri = cv2.arcLength(i, True)
            poly = cv2.approxPolyDP(i, 0.02 * peri, True)
            biggest = poly
            max_area = area_2
    return biggest, max_area


def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew

img = cv2.imread("image.png")
height, width = img.shape[:2]

cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 360, 240)
cv2.createTrackbar("Threshold1", "Trackbars", 0, 255,on_trackbar)
cv2.createTrackbar("Threshold2", "Trackbars", 0, 255,on_trackbar)
cv2.createTrackbar("Clip_w", "Trackbars", 0, width,on_trackbar)
cv2.createTrackbar("Clip_h", "Trackbars", 0, height,on_trackbar)
cv2.createTrackbar("save", "Trackbars", 0, 1,on_trackbar)

while True:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(gray, (5, 5), 1)
    thresh = spliter()
    threshed = cv2.Canny(gauss, thresh[0], thresh[1])
    kernel = np.ones((5,5))
    dilate = cv2.dilate(threshed, kernel, iterations=2)
    erode = cv2.erode(dilate, kernel, iterations=1)

    img_copy = img.copy()
    img_copy_2 = img.copy()

    contours,hierarchy = cv2.findContours(erode,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_copy, contours, -1, (0, 255, 0), 10)
    big,area = bigger(contours)
    try:
        big = reorder(big)
        cv2.drawContours(img_copy,big,-1,(0,0,255),20)
        pts1 = np.float32(big)
        pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img_warp = cv2.warpPerspective(img,matrix,(width,height))
        #
        imgWarp = img_warp[thresh[3]:img_warp.shape[0] - thresh[3], thresh[2]:img_warp.shape[1] - thresh[2]]
        imgWarpColored = cv2.resize(imgWarp, (width, height))
        imgWarpColored = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        cv2.imshow("big", imgWarpColored)

        if cv2.waitKey(1) & 0xFF==27:
            break
        elif thresh[4]==1:

            cv2.imwrite("ok.png", imgWarpColored)
            break
    except:
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        cv2.imshow("big", img_copy)
        if cv2.waitKey(1) & 0xFF==27:
            break
        elif thresh[4]==1:
            cv2.imwrite("ok.png", img_copy)
            break
