# 彩色图 plt.imshow(image[:,:,::-1), 灰度图plt.imshow(image,cmap=plt.cm.gray)
'''
色彩空间转换 convertColor
gray = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
hsv = cv.cvtColor(img1,cv.COLOR_BGR2HSV)
'''

'''图像缩放：对图像进行放大或缩小 res = cv.resize(img1,(2*cols,2*rows),interpolation=cv.INTER_CUBIC) | res1 = cv.resize(img1,None,fx=0.5,fy=0.5)
图像平移：指定平移矩阵后，调用cv.warpAffine()平移图像 M = np.float32([[1,0,100(x)],[0,1,50(y)]]) + dst = cv.warpAffine(img1,M,(cols,rows))
图像旋转：调用cv.getRotationMatrix2D获取旋转矩阵，然后调用cv.warpAffine()进行旋转 M = cv.getRotationMatrix2D((cols/2,rows/2),90,1) 中心+角度+1
仿射变换：调用cv.getAffineTransform(pts1,pts2) (三点)将创建变换矩阵，最后该矩阵将传递给cv.warpAffine()进行变换
透射变换：通过函数cv.getPerspectiveTransform(pts1,pts2) (四点)找到变换矩阵，将cv.warpPerspective()进行投射变换
图像金字塔： cv.pyrUp(): 向上采样 cv.pyrDown(): 向下采样'''

'''
Erosion: img1_erode = cv.erode(thresh1,kernel) #白色部分变少
Dilation: img1_dliate = cv.dilate(thresh1,kernel)
Open: img1_open = cv.morphologyEx(thresh1,cv.MORPH_OPEN,kernel)
Close: img1_close = cv.morphologyEx(thresh1,cv.MORPH_CLOSE,kernel)
Tophat: img1_tophat = cv.morphologyEx(thresh1,cv.MORPH_TOPHAT,kernel_1)
Blackhat: img1_blackhat = cv.morphologyEx(thresh1,cv.MORPH_BLACKHAT,kernel_1)
'''

'''
均值滤波：算法简单，计算速度快，在去噪的同时去除了很多细节部分，将图像变得模糊 cv.blur(img,(5,5))
高斯滤波: 去除高斯噪声 cv.GaussianBlur(img_gaussian,(3,3),25)
中值滤波: 去除椒盐噪声 cv.medianBlur(img, 5)
'''

'''
绘制直方图： cv.calcHist([img1],[0],None,[256],[0,256])
直方图均衡化：增强图像对比度的一种方法 cv.equalizeHist(): 输入是灰度图像，输出是直方图均衡图像
自适应的直方图均衡 clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) + cl1 = clahe.apply(img1)
'''

'''
Sobel算子基于搜索的方法获取边界 (x,y separate 两方向) cv.Sobel(img1, cv.CV_16S, 1, 0)+cv.convertScaleAbs()+cv.addweights()
Laplacian算子基于零穿越获取边界xy = cv.Laplacian(img1, cv.CV_16S) + cv.convertScaleAbs()
Canny算法 canny = cv.Canny(img1, lowThreshold, highThreshold)
'''
''' SIFT_TEMP
img = cv.imread('first_frame.png')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.xfeature2d.SIFT_create()
kp, des = sift.detectAndCompute(gray,None)
cv.drawKeypoints(img, kp, img, flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
'''

# Up to 39