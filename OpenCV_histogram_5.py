# 直方图是对数据进行统计的一种方法，根据灰度图进行绘制的，而不是彩色图像
# 分bins，然后统计像素个数
import cv2 as cv
import matplotlib.pyplot as plt
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

'''研磨mask''' # 使用mask对图像进行遮挡，来控制图像处理的区域
