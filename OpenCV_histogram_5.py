# 直方图是对数据进行统计的一种方法，根据灰度图进行绘制的，而不是彩色图像
# 分bins，然后统计像素个数
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
# 绘制直方图： cv.calcHist([img1],[0],None,[256],[0,256])
# 直方图均衡化：增强图像对比度的一种方法 cv.equalizeHist(): 输入是灰度图像，输出是直方图均衡图像
# 自适应的直方图均衡 clahe = cv.createCLAHE(clipLimit, tileGridSize)

'''直方图'''
# 都要加中括号，注意！
# cv.calcHist(images,channels,mask,histSize,ranges[,hist[,accumulate]])
# channels: 如果输入图像是灰度图，它的值就是 [0]；如果是彩色图像的话，传入的参数可以是 [0]，[1]，[2] 它们分别对应着通道 B，G，R。
# mask: 掩模图像。要统计整幅图像的直方图就把它设为 None。但是如果你想统计图像某一部分的直方图的话，你就需要制作一个掩模图像，并使用它
# histSize:BIN 的数目。也应该用中括号括起来，例如：[256]。
# ranges: 像素值范围，通常为 [0，256]
img1 = cv.imread('first_frame.png',0) #以灰度图的形式导入
hist = cv.calcHist([img1],[0],None,[256],[0,256])
plt.figure(figsize=((10,8)))
plt.plot(hist)
plt.show()

'''研磨mask''' # 使用mask对图像进行遮挡，来控制图像处理的区域,我们这使用mask进行遮挡
mask = np.zeros(img1.shape[:2], np.uint8)
mask[400:600, 200:500] = 255 # RoI
masked_img = cv.bitwise_and(img1, img1, mask=mask) # create the mask on image
mask_hist = cv.calcHist([img1],[0],mask,[256],[0,256])
fig,axes=plt.subplots(nrows=2,ncols=2,figsize=(10,8))
axes[0,0].imshow(img1,cmap=plt.cm.gray)
axes[0,0].set_title("Origin")
axes[0,1].imshow(mask,cmap=plt.cm.gray)
axes[0,1].set_title("蒙版数据")
axes[1,0].imshow(masked_img,cmap=plt.cm.gray)
axes[1,0].set_title("掩膜后数据")
axes[1,1].plot(mask_hist)
axes[1,1].grid()
axes[1,1].set_title("灰度直方图")
plt.show()

'''直方图均衡化''' #将集中的灰度直方图，增加图像的对比度 cv.equallizeHist(img)
img_equalize = cv.equalizeHist(img1)
img_equalize_hist = cv.calcHist([img_equalize],[0],None,[256],[0,256])
plt.imshow(img_equalize,cmap=plt.cm.gray)
plt.show()

plt.plot(img_equalize_hist)
plt.show()

'''自适应直方图均衡化''' #刚才直方图均衡化丢失了一定的信息，因此引入自适应直方图均衡化，将图片分为多块，对每一块进行直方图均衡化
# 设置对比度限制，对于超过限制的灰度值，将其平均分散到其他bins中。最后为了去除每一小块之间的边界，采用双线性插值进行拼接
# cv.createCLAHE(clipLimit (2.0,3.0,40), tileGridSize (8,8))
CLAHE = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = CLAHE.apply(img1) #注意这里需要apply一下
plt.imshow(cl1,cmap=plt.cm.gray)
plt.show()