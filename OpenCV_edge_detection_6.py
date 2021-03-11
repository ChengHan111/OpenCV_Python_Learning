# Sobel, Scharr, laplacian 算子 canny边缘检测
# 边缘检测的目的是标识数字图像中亮度变化明显的点。图像属性中的显著变化通常反映了属性的重要事件和变化
# 边缘检测分为两类：基于搜索(Sobel,Scharr 一阶导数)和基于零穿越(Laplacian二阶导数过零点)
# 当kernel大小为3时，Sobel不准确，因而我们采用Scharr算法，运算结果一样快且更加准确
# Sobel算子基于搜索的方法获取边界 (x,y separate 两方向) cv.Sobel(img1, cv.CV_16S, 1, 0)+cv.convertScaleAbs()+cv.addweights()
# Laplacian算子基于零穿越获取边界xy = cv.Laplacian(img1, cv.CV_16S) + cv.convertScaleAbs()
# Canny算法 canny = cv.Canny(img1, lowThreshold, highThreshold)

import cv2 as cv
import matplotlib.pyplot as plt
'''Sobel''' #Fast, higher efficient than canny, but not very accurate Sobel算子是高斯平滑和微分操作的集合体，抗噪能力强
# Sobel函数求完导数后会有负值，还有会大于255的值。而原图像是uint8，即8位无符号数，所以Sobel建立的图像位数不够，会有截断。
# 因此要使用16位有符号的数据类型，即cv2.CV_16S。处理完图像后，再使用cv2.convertScaleAbs()函数将其转回原来的uint8格式，否则图像无法显示。
# Sobel算子是在两个方向计算的，最后还需要用cv2.addWeighted( )函数将其组合起来
img1 = cv.imread('first_frame.png',0)
# Need to do the sobel to x and y direction in 16
x = cv.Sobel(img1, cv.CV_16S, 1, 0)
y = cv.Sobel(img1, cv.CV_16S,0 , 1)
# Convert back to uint8
Scale_absX = cv.convertScaleAbs(x)
Scale_absY = cv.convertScaleAbs(y)
# addWeighted
result = cv.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
plt.imshow(result,cmap=plt.cm.gray)
plt.show()

'''Scharr (将sobel算子中的ksize设置成-1)'''
x1 = cv.Sobel(img1, cv.CV_16S, 1, 0, ksize=-1)
y1 = cv.Sobel(img1, cv.CV_16S, 0, 1, ksize=-1)
Scale_absX1 = cv.convertScaleAbs(x1)
Scale_absY1 = cv.convertScaleAbs(y1)
result_scharr = cv.addWeighted(Scale_absX1,0.5, Scale_absY1, 0.5, 0)
plt.imshow(result_scharr,cmap=plt.cm.gray)
plt.show()

'''Laplacian''' #二阶导数的边缘检测 laplacian = cv2.Laplacian(src, ddepth[, dst[, ksize[, scale[, delta[, borderType]]]]])
xy = cv.Laplacian(img1, cv.CV_16S)
result_Laplacian = cv.convertScaleAbs(xy)
plt.imshow(result_Laplacian,cmap=plt.cm.gray)
plt.show()

'''Canny''' #最优边缘检测算法
# 1. get rid of the noise through Gaussian Blur
# 2. Sobel
# 3. Non maximum suppression
# 4. 滞后阈值
# canny = cv2.Canny(image, threshold1, threshold2)
# threshold1: minval，较小的阈值将间断的边缘连接起来
# threshold2: maxval，较大的阈值检测图像中明显的边缘
lowThreshold = 0
highThreshold = 100
canny = cv.Canny(img1, lowThreshold, highThreshold)
plt.imshow(canny, cmap=plt.cm.gray)
plt.show()
