# Harris and Shi-Tomasi 角点检测算法具有旋转不变性，但是不具备尺度不变性
# 采用SIFT 算法，提取位置，尺度，旋转不变量
# SIFT的实质是在不同尺度空间上查找关键点，并计算出关键点的方向。
# 我们需要得到图像的高斯金字塔，因为是在不同的尺度下进行的计算，需要不同尺度的图像信息
# 具体看4.3课件
import cv2 as cv
print(cv.__version__)
import matplotlib.pyplot as plt

img = cv.imread('first_frame.png')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# 创建SIFT对象
sift = cv.xfeature2d.SIFT_create()
# 对SIFT对象进行检测，返回关键点和关键点的描述符
kp, des = sift.detectAndCompute(gray,None)
cv.drawKeypoints(img, kp, img, flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
plt.imshow(img[:,:,::-1])
plt.show()
#
# # SIFT在图像的不变特征提取方面具有优势，但是并不完美，仍然存在实用性不高，有时候特征点较少，对边缘光滑的目标无法准确获取特征点等问题
# # SURF算法就是对SIFT算法的改进，计算量小，运算速度快，提取的特征与SIFT几乎相同
img = cv.imread('first_frame.png')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# 创建SIFT对象
sift = cv.xfeature2d.SURF_create(400)
# 对SIFT对象进行检测，返回关键点和关键点的描述符
kp, des = sift.detectAndCompute(gray,None)
cv.drawKeypoints(img, kp, img, flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
plt.imshow(img[:,:,::-1])
plt.show()

'''FAST'''
# 检测器效率很高，比较的仅仅是某个点与周围的差值
# fast = cv.FastFeatureDetector_create(threshold, nonmaxSuppression)
# 注意，利用FAST检测关键点，并没有对应的关键点描述 kp = fast.detect(img, None)
img = cv.imread('tv.jpg')
fast = cv.FastFeatureDetector_create(threshold=38) # 默认NMS开启
kp = fast.detect(img, None)
img2 = cv.drawKeypoints(img, kp, None, color=(0,0,255))
plt.imshow(img2[:,:,::-1])
plt.show()

# 关闭非极大值抑制时
fast.setNonmaxSuppression(0)
kp = fast.detect(img, None)
img3 = cv.drawKeypoints(img, kp, None, color=(0,0,255))
plt.imshow(img3[:,:,::-1])
plt.show()

'''ORB''' # Fast + Brief
# SIFT 和 SURF 受到专利保护，但是ORB不需要，ORB结合了FAST和Brief算法，提出了构造金字塔，为FAST特征点添加了方向，从而
# 使得关键点具有了尺度不变性和旋转不变性
# 实例化ORB orb = cv.xfeatures2d.orb_create(nfeatures) nfeatures为特征点的最大数量
# kp,des = orb.detectAndCompute(gray,None)
img = cv.imread('tv.jpg')
orb = cv.ORB_create(nfeatures=5000)
kp, des = orb.detectAndCompute(img, None)
print(des.shape)
img4 = cv.drawKeypoints(img, kp, None, flags=0)
plt.imshow(img4[:,:,::-1])
plt.show()




