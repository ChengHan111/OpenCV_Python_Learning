# 图像的特征: 角点 角点是图像最重要的特征，对图像图形的理解和分析有很重要的作用
# 角点的特征: 窗口沿任意方向移动都会导致图像灰度的明显变化，将这个思想转换为数学思想时，即将局部窗口向各个方向移动并计算所有灰度
# 差异的总和
# 推导形式见4.2课件
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
'''Harris''' # Harris给出的角点计算方法并不需要计算具体的特征是，而是计算一个角点响应值R来判断角点
# dst=cv.cornerHarris(src, blocksize,ksize,k)
# img：数据类型为 ﬂoat32 的输入图像。
# blockSize：角点检测中要考虑的邻域大小。
# ksize：sobel求导使用的核大小
# k ：角点检测方程中的自由参数，取值参数为 [0.04，0.06].
img = cv.imread('tv.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

gray = np.float32(gray)

dst = cv.cornerHarris(gray, 2, 3, 0.04)
img[dst > 0.001*dst.max()] = [0,0,255]
plt.imshow(img[:,:,::-1])
plt.show()

'''Shi-Tomasi'''
# 对Harris角点检测算法的改进，一般会比Harris算法得到更好的角点
# 只有lamba1和lambda2都大于某个设定的值时，我们才认为他是角点
# corners = cv.goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance)
# Image: 输入灰度图像
# maxCorners : 获取角点数的数目，最多能获得的角点数
# qualityLevel：该参数指出最低可接受的角点质量水平，在0-1之间。
# minDistance：角点之间最小的欧式距离，避免得到相邻特征点。
img = cv.imread('tv.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

corners = cv.goodFeaturesToTrack(gray, maxCorners=1000, qualityLevel=0.01, minDistance=10)
for corner in corners:
    # x, y = corner[0,0], corner[0,1]
    x, y = corner.ravel() #将其拉成一维数组的意思
    cv.circle(img, (x,y), 2, (0,0,255), -1)
plt.imshow(img[:,:,::-1])
plt.show()
