# 平均滤波，高斯滤波，中值滤波 来进行图像去噪 常见的有高斯噪声、椒盐噪声
# 椒盐噪声是一种随机出现的白点或者黑点，可能是亮的区域有黑色像素或是在暗的区域有白色像素（或是两者皆有）
# 高斯噪声是指噪声密度函数服从高斯分布的一类噪声
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
img = cv.imread("salt&pepper.png")
'''图像平滑''' #cv.blur(src, ksize, anchor, borderType)
# 均值滤波：算法简单，计算速度快，在去噪的同时去除了很多细节部分，将图像变得模糊 cv.blur()
# 高斯滤波: 去除高斯噪声 cv.GaussianBlur()
# 中值滤波: 去除椒盐噪声 cv.medianBlur()

'''均值滤波''' #优点算法简单，计算快。 缺点丢失细节，图片变模糊
img_meanfilter = cv.blur(img,(5,5)) #kernel size (5x5),平均5x5图像
# plt.imshow(img_meanfilter[:,:,::-1])
# plt.show()

'''Gaussian blur''' #cv.GaussianBlur(src,ksize,sigmaX,sigmay,borderType)
# ksize 高斯卷积核大小 sigmaX，sigmaY X,Y 方向的标准差，borderType 填充边界类型
img_gaussian = cv.imread('noise_25.png')
img_gaussian_filter = cv.GaussianBlur(img_gaussian,(3,3),25) #这个1是指定X方向的标准差 默认Y方向也跟随X
# plt.imshow(img_gaussian_filter[:,:,::-1])
# plt.show()

'''Median filter''' #Useful for salt&pepper noise 因为椒盐噪声是突然的值变化，采用中值滤波可以显著消除这种突变
# cv.medianBlur(src, ksize)
img_median = cv.medianBlur(img, 5 )
plt.imshow(img_median[:,:,::-1])
plt.show()




