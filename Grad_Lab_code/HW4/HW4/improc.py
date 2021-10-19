from __future__ import division
import cv2 #OpenCV
import matplotlib.pyplot as plt
from PIL import Image #pillow
import numpy as np

#cv2; imread, imshow (compare with plt) cvtColor, filter2D, Canny, Sobel
img=cv2.imread('.\cat.jpg') #imports as BGR, NOT RGB!; set filepath to your own
imgc=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #convert to RGB
imgc1=np.flip(img,axis=2) #also works
img_g=cv2.cvtColor(imgc,cv2.COLOR_RGB2GRAY) #grayscale
img_hsv=cv2.cvtColor(imgc,cv2.COLOR_RGB2HSV) #HSV
img_lab=cv2.cvtColor(imgc,cv2.COLOR_RGB2LAB) #LAB
box=np.ones((20,20))/400 #Box filter kernel, essentially an averager
img_filt=cv2.filter2D(imgc,-1,np.flip(box))#MUST FLIP KERNEL FOR CONVOLUTION; if no flip, it's correlation
img_canny=cv2.Canny(imgc,150,200) #edge detector
img_sobelx=cv2.Sobel(img_g,-1,1,0) #x-derivative
img_sobely=cv2.Sobel(img_g,-1,0,1) #y-derivative
img_sobel=cv2.Sobel(img_g,-1,1,1) #both
cv2.imshow('img_sobelx',img_sobelx)

#matplotlib
plt.figure()
plt.imshow(img[:,:,2],cmap='gray') #equivalently, cv2.imread('Grad_Python\\cat.jpg',2)
plt.xticks([])  #or use set_xticks if axis object or subplot
plt.yticks([]) #same as above
plt.title('Cat')

#linear contrast stretching
img_stretch=255*(img_sobel-np.amin(img_sobel))/(np.amax(img_sobel)-np.amin(img_sobel))

#PIL, Image module, open, resize (& resample), rotate
cat=Image.open('.\cat.jpg') #load images in RGB
cat_arr=np.array(cat) #convert PIL image to usable array form
cat_r=cat.resize((100,100),resample=Image.BILINEAR)
plt.figure()
plt.imshow(np.array(cat_r))
cat_rot=cat.rotate(30) #rotates counterclockwise by default, negative for clockwise

#gamma, alpha
cat_float=cat_arr.astype('float64') #Images typically imported as uint8, CONVERT TO FLOAT BEFORE MATHEMATICAL OPERATIONS!
cat_gamma_2=255*((cat_float**2)-np.amin(cat_float**2))/(np.amax(cat_float**2)-np.amin(cat_float**2))
cat_gamma_05=255*((cat_float**0.5)-np.amin(cat_float**0.5))/(np.amax(cat_float**0.5)-np.amin(cat_float**0.5))
plt.figure()
plt.imshow(cat_gamma_05.astype('uint8')) #CONVERT BACK TO uint8 BEFORE SHOWING
dog=Image.open('.\cat.jpg')
dog_r=dog.resize((np.shape(cat)[1],np.shape(cat)[0])) #MAKE SURE SIZES OF ALPHA BLEND IMAGES ARE SAME
dog_float=np.array(dog_r).astype('float64')
dog_cat_blend=0.7*cat_float+(1-0.7)*dog_float