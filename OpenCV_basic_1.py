import numpy as np
import cv2 as cv
import matplotlib.pyplot  as plt

img = cv.imread('first_frame.png',1) #1, 0 ,-1 for color, grayscale and alpha

# Get the shape of the image
print(img.shape)
# Get the type of the image
print(img.dtype)
# Get the size of the image
print(img.size) # length * height * #channels

# Split R, G, B channels
b, g, r = cv.split(img)
# Merge channels we can now get back the image
img = cv.merge((b,g,r))

# paramaters: the name of the window, the img to what to show
# cv.imshow('test1',img)
# cv.waitKey(0)

# Notice that we show do an inverse since opencv saves image in the order of BGR, while we should save plt as RGB
plt.imshow(img[:,:,::-1])
plt.imshow(img,cmap=plt.cm.gray)
# plt.show()

# paramaters: the name of the saving file, the img want to save
cv.imwrite("test1_saved.png", img)

# Draw a line: cv.line(img,start,end,color,thickness)
# Draw a circle: cv.circle(img,centerpoint, r, color, thickness)
# Draw a rectangle: cv.rectangle(img, leftupper, rightdown, color, thickness)
# Put text: cv.putText(img,text,station,font,fontsize,color,thickmess,cv.LINE_AA)

# 1 创建一个空白的图像, pure black image
img = np.zeros((512,512,3), np.uint8) #This also create an image with BGR, when using plt.imshow, remember to inverse
# 2 绘制图形
cv.line(img,(0,0),(511,511),(255,0,0),5)
cv.rectangle(img,(384,0),(510,128),(0,255,0),3)
# In cv.circle, -1 in the last part means we have a fulfill circle, if we are using numbers > 0, it means thickness
cv.circle(img,(447,63), 63, (0,0,255), -1)
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv.LINE_AA)
# 3 图像展示
plt.imshow(img[:,:,::-1])
plt.title('result'), plt.xticks([]), plt.yticks([])
plt.show()


# Get and change some pixels in the image. For color image, we return R, G, B values. For grayscale image, one value is returned.
px = img[100,100] #find specific pixel's value
#get the value of blue channel
img[100,100,0] # in the form of BGR, so B is the first form in this img
# change value
img[100,100] = [255,255,255]

# Most common color space changed methods:
# BGR to Gray and BGR to HSV
img1 = cv.imread("first_frame.png")
plt.imshow(img1[:,:,::-1])
# b, g, r = cv.split(img1)
# plt.imshow(b,cmap=plt.cm.gray)
# plt.show()

gray = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
plt.imshow(gray, cmap=plt.cm.gray)
plt.show()

hsv = cv.cvtColor(img1,cv.COLOR_BGR2HSV)
plt.imshow(hsv)
plt.show()

# Add on images (same shape or the second image is a 标量)
# We can use Opencv or numpy for the calculation, by numpy uses mod while Opencv reaches 255 when add calculation gets the result
# > 255, we suggest Opencv here.
# Add img1 and img2, remember always flip when we use plt
img3 = cv.add(img1, img2)

# Mix on images (which is also an add calculation with weights)
# alpha*img1 + (1 - alpha)*img2 we can use cv.addWeighted() for mixture
img3 = cv.addWeighted(img1, 0.7, img2, 0.3, 0) # the last value is gamma




