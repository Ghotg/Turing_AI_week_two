import cv2 as cv

img = cv.imread("image.png")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow("image", img)
if cv.waitKey(0) & 0xFF==27:
    cv.destroyAllWindows()
