# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 23:50:18 2021

@author: wgrim
"""

from __future__ import division
#from numpy import * (functions without prefix)
#import numpy (numpy.function)
import numpy as np #(np.function, typical way done)

if __name__=='__main__':
    #for loop to repeat a block of code over a specific range of values
    x=0
    for i in range(0,20,1): #range(20)
       # x=x+i
        x+=i
        x-=i
        
    #while loops until a given condition is met
    while x<20:
        x+=1
        
    #lambda keyword creates fruitful functions with one line
    func=lambda x,y: x*y
    
    #more list methods:
    mylist=[1,True,'Grad Lab',[1,2]]
    #use % for modulus/remainder, count to count the occurences of a given 
    #element in a list, max and min for maximum and minimum
    print(3%2,mylist.count('Grad Lab'))
    print(max([1,2]),min([1,2]))
    
    #use numpy.array to create arrays, vectors, matrices
    arr=np.array([1,2,3])
    mat=np.array([[1,2,3],[4,5,6]])
    
    #np.arange creates arrays with specific step sizes, minimum (inclusive)
    #and maximum (exclusive) values
    ara=np.arange(1,10,1)
    ara2=np.arange(1,100,2)
    for i in np.arange(0,20,1): #range(20)
       # x=x+i
        x+=i
    
    #np.linspace creates arrays with both endpoints included and length specified
    lin=np.linspace(1,10,9) #linspace是包括最后的点的
    print('lin:',lin)
    
    #zeros creates arrays entirely of zeros
    z0=np.zeros(10)
    z1=np.zeros((10,10))
    print(z1+2)
    
    #ones creates arrays entirely of ones
    o=np.ones((10,10))
    print(o*2)
    
    #empty creates an empty array
    empty=np.empty(3)
    #must define array in advance before using loop to include numbers in it
    newarr=np.empty(10)
    for i in np.arange(0,10,1):
        newarr[i]=i*5
        
    #where function can locate indexes where array values satisfy a given
    #condition; these can mask an array such that only those values are returned
    mask=np.where(newarr<=20)
    
    #random module can be used to generate random arrays
    randarr=np.random.randint(1,10)
    sd=np.random.randint(0,2**32-1,dtype='int64') #strategy to ensure repeatability of random analysis
    print(sd)
    np.random.seed(sd)
    random_arr=np.random.randn(2,3)
    rar=np.random.randint(1,10,(5,5))
    r_uni=np.random.uniform(0,1,3) #many distributions can be used
    
    #attributes can be used to retrieve values corresponding to arrays
    rarspace=rar.itemsize*rar.size #gives size of data in bytes
    
    #transpose arrays like this
    rarT1=np.transpose(rar)
    rarT2=rar.T
    
    #max/min can be used with numpy too, can find indexes where they are as well
    rar_maxarg=np.argmax(rar)
    rar_maxarg=np.where(rar==np.amax(rar))
    
    #can calculate mean and sum of arrays
    print(np.mean(rar),np.sum(rar))
    
    #eigenvalues and eigenvectors calculated using this function
    eigs=np.linalg.eig(rar)
    #U,e,Vt=np.linalg.svd(rar), remember this for IPCV!
    
    #reshape can alter the shape of the array, but make sure the same number of
    #elements are used in the reshaping
    arr1=np.arange(1,10,1)
    arr1=np.reshape(arr1,(3,3))
    arr2=np.arange(2,11,1)
    arr2=np.reshape(arr2,(3,3))
    
    #Hadamard, or element wise multiplication:
    arr_had=arr1*arr2

    #Matrix multiplication:
    arr_dot1=np.dot(arr1,arr2)
    arr_dot2=arr1@arr2
    
    #add vector functions together like this:
    adds=np.array([1,2])+np.array([3,4])
    adds2=(np.array([[1,2],[3,4]]).T+np.array([3,4])).T
    
    #append can append a number or array to another array
    arr3=np.append(np.arange(1,10,1),20)
    adds3=np.append(adds2.T,np.array([[3,4]]),axis=0).T
    newarr=np.array([])
    for i in np.arange(0,10,1):
        newarr=np.append(newarr,i*5)