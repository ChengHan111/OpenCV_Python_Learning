# Harris and Shi-Tomasi 角点检测算法具有旋转不变性，但是不具备尺度不变性
# 采用SIFT 算法，提取位置，尺度，旋转不变量
# SIFT的实质是在不同尺度空间上查找关键点，并计算出关键点的方向。
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('first_frame.png')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.xfeature2d.SIFT_create()
kp, des = sift.detectAndCompute(gray,None)
cv.drawKeypoints(img, kp, img, flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
plt.imshow(img[:,:,::-1])
plt.show()
