# OpenCV 进行线和圆的检测
# 所谓的模板匹配，就是在给定的图片中查找和模板最相似的区域，是按照滑动窗口的思路计算匹配度
# res = cv.matchTemplate(img, template, method) 其输出结果是一个二值矩阵(0,1)
'''
匹配方法：
平方差匹配(CV_TM_SQDIFF)：利用模板与图像之间的平方差进行匹配，最好的匹配是0，匹配越差，匹配的值越大。
相关匹配(CV_TM_CCORR)：利用模板与图像间的乘法进行匹配，数值越大表示匹配程度较高，越小表示匹配效果差。
利用相关系数匹配(CV_TM_CCOEFF)：利用模板与图像间的相关系数匹配，1表示完美的匹配，-1表示最差的匹配。
完成匹配后，使用cv.minMaxLoc()方法查找最大值所在的位置即可。如果使用平方差作为比较方法，则最小值位置是最佳匹配位置。
'''
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('wulin.jpeg')
template = cv.imread('bai.jpeg')
h,w,l = template.shape

res = cv.matchTemplate(img, template, cv.TM_CCORR)
plt.imshow(res,cmap=plt.cm.gray)
plt.show()
# 返回最匹配的位置，确定左上角坐标
# 最小值，最大值，最小位置，最大位置 minMaxLoc函数找到最小值和最大值元素值以及它们的位置
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
top_left = max_loc #when we use CV_TM_SQDIFF, we use top_left = min_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv.rectangle(img, top_left, bottom_right, (0,255,0), 2) #(0, 255, 0) means color

plt.imshow(img[:,:,::-1])
plt.title('result'), plt.xticks([]),plt.yticks([])
plt.show()

'''
但是模板匹配存在问题，如果尺度发生变换，视角发生变化。那么模板匹配就不适用了，适用方法为(SIFT&SURF etc）
上述两种方法的主要思路是首先通过关键点检测算法获取模板和测试图片中的关键点，然后使用关键点匹配算法处理。
这些关键点可以很好的处理尺度变换，势角变换，旋转辩护那，光照变化等，具有很好的不变性 (之后介绍)
'''


'''Hough Transform Lines''' #cv.HoughLines(img, rho, theta, threshold)
# 可以提取图像中的直线，圆等几何形状
# Hough Space 笛卡尔坐标系中的一条直线，对应Hough Space中的一个点
# 如果在笛卡尔坐标系中的点共线，那么这些点在霍夫空间中对应的直线交于一点
#  当存在很多点时，我们尽可能选择多的直线汇成的点(在Hough Space上)
# 我们还要转换成极坐标系，因为直角坐标系下在Hough Space中平行或垂直的线无法表示
# r = xcos@ + ysin@
# 此时霍夫空间不再是(k,b)空间了，而是变成了(r,@)空间，但是同理，在hough space中过同一个点就共线
# cv.HoughLines(img, rho, theta, threshold)
# img: 二值化图像(在调用霍夫变换前先要进行二值化，或者先进行Canny边缘检测)
# rho, theta r和@的精确度
# threshold 只有累加器中的值高于阈值时才被认定为直线

import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('Crossroad_Sunrise.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 通过调整Canny和HoughLines的threshhold可以获得较好的检测效果
edges = cv.Canny(gray, 90, 150)
plt.imshow(edges,cmap=plt.cm.gray)
plt.show()
# 霍夫直线变换
lines = cv.HoughLines(edges, 0.8, np.pi/180, 200) # np.pi/180 means 每一度作为精度范围
# print(lines)
# 将检测的线绘制在图像上(注意是极坐标系)
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    # 将线延长
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv.line(img, (x1, y1), (x2, y2), (0, 255, 0))

plt.figure(figsize=(10,8),dpi=100)
plt.imshow(img[:,:,::-1]),plt.title('Hough Transform Line')
plt.xticks([]), plt.yticks([])
plt.show()

'''Hough Transform Circle''' # circles = cv.HoughCircles(image, method, dp, minDist, param1=100, param2=100, minRadius=0, maxRadius=100)
# 因为圆的表达式含三个参数: (x,y)坐标以及半径，如果直接霍夫圆检测就是在三个参数组成的三维空间累加器上进行圆形检测，效率很低
# OpenCV中采用霍夫梯度法进行圆形的检测
# 原则上霍夫变换可以检测任何形状，但是复杂形状需要的参数就越多。霍夫梯度法是霍夫变换的改进，他的目的是减小霍夫空间的维度，提高效率
''' circles = cv.HoughCircles(image, method, dp, minDist, param1=100, param2=100, minRadius=0, maxRadius=100)
参数：
image：输入图像，应输入灰度图像
method：使用霍夫变换圆检测的算法，它的参数是CV_HOUGH_GRADIENT
dp：霍夫空间的分辨率，dp=1时表示霍夫空间与输入图像空间的大小一致，dp=2时霍夫空间是输入图像空间的一半，以此类推
minDist为圆心之间的最小距离，如果检测到的两个圆心之间距离小于该值，则认为它们是同一个圆心
param1：边缘检测时使用Canny算子的高阈值，低阈值是高阈值的一半。
param2：检测圆心和确定半径时所共有的阈值
minRadius和maxRadius为所检测到的圆半径的最小值和最大值
返回：circles：输出圆向量，包括三个浮点型的元素——圆心横坐标，圆心纵坐标和圆半径
'''
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

planets = cv.imread('asteroids.jfif')
gray_img = cv.cvtColor(planets, cv.COLOR_BGR2GRAY)
# 由于霍夫圆检测对噪声比较敏感，因此首先要对图像进行中值滤波去除噪点
# img_equalize = cv.equalizeHist(gray_img)
img = cv.medianBlur(gray_img, 7)
plt.imshow(img,cmap=plt.cm.gray)
plt.show()
# Hough gradient 霍夫梯度运算
circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 15, param1=90, param2=20, minRadius=0, maxRadius=100)
# print(circles)
for i in circles[0,:]:
    # 绘制圆形
    cv.circle(planets, (i[0],i[1]), i[2], (0,255,0), 2)
#     绘制圆心
    cv.circle(planets, (i[0],i[1]), 2, (0,0,255), 3)
plt.figure(figsize=(10,8),dpi=100)
plt.imshow(planets[:,:,::-1]), plt.title('Hough Transform Circle')
plt.xticks([]), plt.yticks([])
plt.show()
    





