#cat image courtesy of https://en.wiktionary.org/wiki/cat
#dog image courtesy of:
#https://www.abcactionnews.com/news/region-hillsborough/tampa-ranked-the-best-city-in-the-country-to-be-a-dog
from __future__ import division
import cv2 #OpenCV
import matplotlib.pyplot as plt
from PIL import Image #pillow
import numpy as np


#1) Load in the provided cat image using OpenCV (remember to convert to RGB); 
#use an x Sobel filter on the red channel, a y Sobel filter on the blue 
#channel (both with a filter size of k=5), and a Canny filter on the green 
#channel (test different threshold combinations for a good edge image), 
#and show the results via matplotlib in side-by-side subplots
#cv2; imread, imshow (compare with plt) cvtColor, filter2D, Canny, Sobel
img=cv2.imread('.\cat.jpg') #imports as BGR, NOT RGB!; set filepath to your own
imgc=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #convert to RGB
img_sobelx=cv2.Sobel(imgc[:,:,0],-1,1,0,ksize=5) #x-derivative
img_sobely=cv2.Sobel(imgc[:,:,2],-1,0,1,ksize=5) #y-derivative
img_canny=cv2.Canny(imgc[:,:,1],150,200) #edge detector

plt.figure(figsize=(10,5))
plt.suptitle('Sobel_and_Canny')
plt.subplot(1,3,1), plt.title('img_sobelx')
plt.imshow(img_sobelx.astype('uint8')), plt.axis('off')
plt.subplot(1,3,2), plt.title('img_sobely')
plt.imshow(img_sobely.astype('uint8')), plt.axis('off')
plt.subplot(1,3,3), plt.title('img_canny')
plt.imshow(img_canny.astype('uint8')), plt.axis('off')
plt.show()



#2) Now convert the cat image to grayscale, then create a 3x3 Gaussian 
#filter with mean zero and standard deviation 2, use it to blur the 
#gray cat image, and show the result
img_g=cv2.cvtColor(imgc,cv2.COLOR_RGB2GRAY) #grayscale
# apply guassian blur on src image
# dst = cv2.GaussianBlur(img_g, (3, 3), 0, 2)
img_blur = cv2.GaussianBlur(img_g, (3,3), 2, None, 2)
# display input and output image
cv2.imshow("Gaussian Smoothing", img_blur)
cv2.waitKey(0)  # waits until a key is pressed


#3) Use PIL to load the same image (in RGB); halve the size of the image, 
#resample it using bilinear interpolation, and rotate it counterclockwise 
#by 60 degrees
cat=Image.open('.\cat.jpg') #load images in RGB
cat_arr=np.array(cat) #convert PIL image to usable array form
cat_r=cat.resize((1280,596),resample=Image.BILINEAR) # This is the half size of the image
cat_rot=cat.rotate(60)
plt.figure()
plt.imshow(np.array(cat_rot))
plt.show()


#4) Gamma correct the RGB cat image (use either OpenCV or PIL as desired)
#using gamma values of 0.5 and 2, then show the results in side-by-side
#subplots
cat_float=cat_arr.astype('float64') #Images typically imported as uint8, CONVERT TO FLOAT BEFORE MATHEMATICAL OPERATIONS!
cat_gamma_2=255*((cat_float**2)-np.amin(cat_float**2))/(np.amax(cat_float**2)-np.amin(cat_float**2))
cat_gamma_05=255*((cat_float**0.5)-np.amin(cat_float**0.5))/(np.amax(cat_float**0.5)-np.amin(cat_float**0.5))
plt.figure(figsize=(10,5)) #设置窗口大小
plt.suptitle('Cat_Gamma_Different_Value') # 图片名称
plt.subplot(1,2,1), plt.title('cat_gamma_0.5')
plt.imshow(cat_gamma_05.astype('uint8')), plt.axis('off')
plt.subplot(1,2,2), plt.title('cat_gamma_2')
plt.imshow(cat_gamma_2.astype('uint8')), plt.axis('off') #这里显示灰度图要加cmap
plt.show()



#5) Load in the dog image in RGB, use PIL to resize it to the same size as 
#the cat image, and alpha blend the images together into one image, with 
#the new image making up 70% and the cat making 30%
dog=Image.open('.\dog.jpg')
dog_r=dog.resize((np.shape(cat)[1],np.shape(cat)[0])) #MAKE SURE SIZES OF ALPHA BLEND IMAGES ARE SAME
dog_float=np.array(dog_r).astype('float64')
dog_cat_blend=0.7*cat_float+(1-0.7)*dog_float
plt.figure()
plt.imshow(np.array(dog_cat_blend.astype('uint8')))
plt.show()

