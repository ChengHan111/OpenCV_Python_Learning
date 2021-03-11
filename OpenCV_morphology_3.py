# 腐蚀，膨胀，开闭运算，礼帽和黑帽
# 邻域有：4邻域(cross)，D邻域(4个对角线)，8邻域， 与之相对应的有四联通，八联通和m联通。m联通条件特殊
# 形态学操作通常在二进制图像中进行
# 腐蚀膨胀都是针对白色部分，膨胀是白色部分扩张，腐蚀是白色部分收缩
# 腐蚀是与操作，全1为1，否则为零
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
'''Erosion & Dilation'''
# cv.erode(img,kernel,iterations) 其中iter是腐蚀的次数
# cv.dilate(img,kernel,iterations)
img1 = cv.imread("first_frame.png",0)
ret,thresh1 = cv.threshold(img1,127,255,cv.THRESH_BINARY) #cv.THRESH_BINARY_INV 取反
cv.imshow('image_origin_binary',thresh1)
kernel = np.ones((5,5), np.uint8)
img1_erode = cv.erode(thresh1,kernel) #白色部分变少
img1_dliate = cv.dilate(thresh1,kernel)
# cv.imshow('image_erode',img1_erode)
# cv.imshow('image_dliate',img1_dliate)
# cv.waitKey(0)

'''Open & Close operation''' #将腐蚀和膨胀按照一定的顺序处理
# Open: 开运算是先腐蚀后膨胀，其作用是：分离物体，消除小区域。特点：消除噪点，去除小的干扰块，而不影响原来的图像。
# Close: 先膨胀后腐蚀，作用是消除/“闭合”物体里面的孔洞，特点：可以填充闭合区域。 底色为黑，前色为白时！
# cv.morphologyEx(img, op, kernel) op: 处理方式：若进行开运算，则设为cv.MORPH_OPEN，若进行闭运算，则设为cv.MORPH_CLOSE 

img1_open = cv.morphologyEx(thresh1,cv.MORPH_OPEN,kernel)
img1_close = cv.morphologyEx(thresh1,cv.MORPH_CLOSE,kernel)
# cv.imshow('img1_open',img1_open)
# cv.imshow('img1_close',img1_close)
# cv.waitKey(0)

'''Top hat & Black hat (礼帽与黑帽)'''
# Top hat: 原图与开运算的结果之差突出了比原图轮廓周围的区域更明亮的区域
# 顶帽运算往往用来分离比邻近点亮一些的斑块。当一幅图像具有大幅的背景的时候，而微小物品比较有规律的情况下，可以使用顶帽运算进行背景提取。
# Black hat: 闭运算与原图的结果之差 突出了比原图轮廓周围的区域更暗的区域
# 黑帽运算用来分离比邻近点暗一些的斑块。非常完美的轮廓效果图
# cv.MORPH_BLACKHAT and cv.MORPH_TOPHAT
kernel_1 = np.ones((10,10), np.uint8)
img1_tophat = cv.morphologyEx(thresh1,cv.MORPH_TOPHAT,kernel_1)
img1_blackhat = cv.morphologyEx(thresh1,cv.MORPH_BLACKHAT,kernel_1)
cv.imshow('img1_tophat',img1_tophat)
cv.imshow('img1_blackhat',img1_blackhat)
cv.waitKey(0)