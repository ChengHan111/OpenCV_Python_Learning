import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
'''图像缩放：对图像进行放大或缩小cv.resize()
图像平移：指定平移矩阵后，调用cv.warpAffine()平移图像
图像旋转：调用cv.getRotationMatrix2D获取旋转矩阵，然后调用cv.warpAffine()进行旋转
仿射变换：调用cv.getAffineTransform将创建变换矩阵，最后该矩阵将传递给cv.warpAffine()进行变换
透射变换：通过函数cv.getPerspectiveTransform()找到变换矩阵，将cv.warpPerspective()进行投射变换
图像金字塔： cv.pyrUp(): 向上采样 cv.pyrDown(): 向下采样'''

'''图像缩放'''
# cv2.resize(src,dsize,fx=0,fy=0,interpolation=cv2.INTER_LINEAR)
# src : 输入图像
# dsize: 绝对尺寸，直接指定调整后图像的大小
# fx,fy: 相对尺寸，将dsize设置为None，然后将fx和fy设置为比例因子即可
# interpolation：插值方法，当图像进行放大或者缩小时，会有新的像素产生，使用插值方法产生新像素 (4种插值方法，默认INTER_AREA)
img1 = cv.imread("first_frame.png")
rows,cols = img1.shape[:2]
# 2.1 绝对尺寸
res = cv.resize(img1,(2*cols,2*rows),interpolation=cv.INTER_CUBIC) #注意先col 后row
# 2.2 相对尺寸
res1 = cv.resize(img1,None,fx=0.5,fy=0.5)
# 3 图像显示
# 3.1 使用opencv显示图像(不推荐)
# cv.imshow("orignal",img1)
# cv.imshow("enlarge",res)
# cv.imshow("shrink）",res1)
# cv.waitKey(0)

# 3.2 使用matplotlib显示图像
fig,axes=plt.subplots(nrows=1,ncols=3,figsize=(10,8),dpi=100)
axes[0].imshow(res[:,:,::-1])
axes[0].set_title("absolute")
axes[1].imshow(img1[:,:,::-1])
axes[1].set_title("origin")
axes[2].imshow(res1[:,:,::-1])
axes[2].set_title("shrink(not absolute)")
plt.show()

'''图像平移'''
# cv.warpAffine(img,M,dsize) where M is a (2x3) numpy array, dsize is the size of the output image
rows,cols = img1.shape[:2]
M = M = np.float32([[1,0,100],[0,1,50]])# 平移矩阵 100在x方向, 50在y方向
dst = cv.warpAffine(img1,M,(cols,rows)) #注意先col 后row

fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(10,8),dpi=100)
axes[0].imshow(img1[:,:,::-1])
axes[0].set_title("Origin")
axes[1].imshow(dst[:,:,::-1])
axes[1].set_title("AfterAffine")
plt.show()

'''图像旋转''' #cv2.getRotationMatrix2D(center, angle, scale)
# 通过getRotationMatrix获得旋转矩阵，再输入到上面的平移矩阵函数warpAffine中获得旋转后的图像
rows,cols = img1.shape[:2]
# 2.1 生成旋转矩阵
M = cv.getRotationMatrix2D((cols/2,rows/2),90,1)
# 2.2 进行旋转变换
dst = cv.warpAffine(img1,M,(cols,rows))
plt.imshow(dst[:,:,::-1])
plt.show()

'''图像仿射变换''' #图像的形状位置角度的变化 getAffineTransform
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[100,100],[200,50],[100,250]])

M = cv.getAffineTransform(pts1,pts2) #pts1图像中的三个点对应pts2中图像中的三个点，通过三点确定仿射矩阵M
dst = cv.warpAffine(img1,M,(cols,rows))
plt.imshow(dst[:,:,::-1])
plt.show()

'''图像透射变换''' #从透视中心，像点，目标点三点共线的原理。更换投影中心，投影图像到新的视平面
pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]]) # we should have 4 points
pts2 = np.float32([[100,145],[300,100],[80,290],[310,300]])
# 透射变换矩阵计算
T = cv.getPerspectiveTransform(pts1,pts2)
# 2.2 进行变换 Note that we are using cv.warpPerspective here
dst = cv.warpPerspective(img1,T,(cols,rows))
plt.imshow(dst[:,:,::-1])
plt.show()

'''图像金字塔，upsample and downsample''' #上采样和下采样
up_img = cv.pyrUp(img1)  # 上采样操作
down_img = cv.pyrDown(img1)  # 下采样操作
# cv.imshow('enlarge', up_img)
# cv.imshow('original', img1)
# cv.imshow('shrink', down_img)
# cv.waitKey(0)
# cv.destroyAllWindows()









