# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 23:34:39 2021

@author: wgrim
"""

from __future__ import division
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

if __name__=='__main__':
    #cat image courtesy of https://en.wiktionary.org/wiki/cat
    #dog image courtesy of 
    #https://www.abcactionnews.com/news/region-hillsborough/tampa-ranked-the-best-city-in-the-country-to-be-a-dog
    #1) Load in the provided cat image using OpenCV (remember to convert to RGB); 
    #use an x Sobel filter on the red channel, a y Sobel filter on the blue 
    #channel (both with a filter size of k=5), and a Canny filter on the green 
    #channel (test different threshold combinations for a good edge image), 
    #and show the results via matplotlib in side-by-side subplots
    cat=cv2.imread(r'C:\Users\hanch\Downloads\Grad_Lab_code\HW4\HW4\cat.jpg')
    fig,ax=plt.subplots(1,3)
    ax[0].set_title('Cat, Red, Sobel x')
    ax[0].imshow(cv2.Sobel(cat[:,:,2],-1,1,0,ksize=5),cmap='gray')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_title('Cat, Blue, Sobel y')
    ax[1].imshow(cv2.Sobel(cat[:,:,0],-1,0,1,ksize=5),cmap='gray')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[2].set_title('Cat, Green, Canny')
    ax[2].imshow(cv2.Canny(cat[:,:,1],150,180),cmap='gray')
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    #2) Now convert the cat image to grayscale, then create a 3x3 Gaussian 
    #filter with mean zero and standard deviation 2, use it to blur the 
    #gray cat image, and show the result
    graycat=cv2.cvtColor(cat,cv2.COLOR_BGR2GRAY)
    x=np.arange(0,2*1+1,1)-1
    xx,yy=np.meshgrid(x,x)
    r2=xx**2+yy**2
    g0=np.exp(-r2/(2*2**2))
    gauss=g0/np.sum(g0)
    blur_cat=cv2.filter2D(graycat,-1,np.flip(gauss))
    plt.figure()
    plt.title('Cat, Gaussian Blurred')
    plt.imshow(blur_cat.astype('uint8'),cmap='gray')
    plt.xticks([])
    plt.yticks([])
    #3) Use PIL to load the same image (in RGB); halve the size of the image, 
    #resample it using bilinear interpolation, and rotate it counterclockwise 
    #by 60 degrees
    cat_pil=Image.open(r'C:\Users\hanch\Downloads\Grad_Lab_code\HW4\HW4\cat.jpg')
    ncat=cat_pil.resize((int(np.shape(cat_pil)[1]/2),int(np.shape(cat_pil)[0]/2))\
                        ,resample=Image.BILINEAR)
    rncat=ncat.rotate(60)
    plt.figure()
    plt.title('Cat, Rotated and Resized')
    plt.imshow(rncat)
    plt.xticks([])
    plt.yticks([])
    #4) Gamma correct the RGB cat image (use either OpenCV or PIL as desired)
    #using gamma values of 0.5 and 2, then show the results in side-by-side
    #subplots
    lin_cont_stretch=lambda img: 255*(img-np.amin(img))/(np.amax(img)-np.amin(img))
    gamma=lambda img, g: lin_cont_stretch(img**g)
    cat=cv2.cvtColor(cat,cv2.COLOR_BGR2RGB)
    fig,ax=plt.subplots(1,2)
    ax[0].set_title('Cat, Gamma=0.5')
    ax[0].imshow(gamma(cat.astype('float64'),0.5).astype('uint8'))
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_title('Cat, Gamma=2')
    ax[1].imshow(gamma(cat.astype('float64'),2).astype('uint8'))
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    #5) Load in the dog image in RGB, use PIL to resize it to the same size as 
    #the cat image, and alpha blend the images together into one image, with 
    #the new image making up 70% and the cat making 30%
    dog0=Image.open(r'C:\Users\hanch\Downloads\Grad_Lab_code\HW4\HW4\dog.jpg')
    dogP=dog0.resize((int(np.shape(cat_pil)[1]),int(np.shape(cat_pil)[0]))\
                        ,resample=Image.BILINEAR)
    dog=np.array(dogP)
    blend=lin_cont_stretch(0.7*dog+0.3*cat)
    plt.figure()
    plt.title('Dog and Cat Blend')
    plt.imshow(blend.astype('uint8'))
    plt.xticks([])
    plt.yticks([])