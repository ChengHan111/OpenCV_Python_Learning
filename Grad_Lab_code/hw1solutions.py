# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 11:41:17 2021

@author: wgrim
"""

from __future__ import division
import numpy as np

if __name__=='__main__':
    #1) Print your name
    print('William Grimble')
    #2) Use lambda keyword to write a function 'func' that squares an input 
    #and adds 5 to it, then calculate it with x=3 and print the result
    func=lambda x: x**2+5
    print(func(3))
    #3) Use def keyword to create a function 'newfunc' that returns the input 
    #squared plus 5 if less than 5 (can use 'func' if desired) and multiplies 
    #it by 12 otherwise, then calculate with x=3 and x=7 and print the results
    def newfunc(x):
        if x<5:
            return func(x)
        else:
            return x*12
    print(newfunc(3))
    print(newfunc(7))
    #4) Make a list 'foo' with the first element being the string 'Grad Lab', 
    #the second being the boolean True, the third being a list consisting of 
    #integer 2 and float 4, then print it
    foo=['Grad Lab', True, [2,4.]]
    print(foo)
    #5) Append the float number 1 to the end of foo, then delete the boolean 
    #in index 1, then print the result
    foo.append(1.)
    foo.pop(1)
    print(foo)
    #6) Make a tuple 'too' using the same initial elements of 'foo' along with 
    #the floating 1 at the end and print it 
    too=('Grad Lab', True, [2,4.],1)
    print(too)
    #7) Print the second and third elements of both 'foo' and 'too' using 
    #index slicing
    print(foo[1:3],too[1:3])
    #8) Print the odd elements of both using index slicing
    print(foo[1::2],too[1::2])
    #9) Create a dictionary 'course' with 'class' as the key for 'Grad Lab', 
    #'taking' as the key for True, 'times' as the key for the 2 and 4 list, and 
    #'assignment' as the key for 1, then print any one result by calling the
    #corresponding key from the dictionary
    course={'class':'Grad Lab', 'taking':True, 'times':[2,4.],'assignment':1}
    print(course['class'],course['taking'],course['times'],course['assignment'])
    #10) Use a for loop to calculate 0+1+2+..+100 and print the result
    x=0
    for i in range(101):
       x+=i
    print(x)
    #11) Create a Numpy array 'npi' consisting of the numbers 3,5,15,10,11, 
    #and print both the array and its maximum value
    npi=np.array([3,5,15,10,11])
    print(npi,np.amax(npi))
    #12) Make an array 'ara' going from 2 to 50 in increments of 2 and print it
    ara=np.arange(2,52,2)
    print(ara)
    #13) Make an array 'lin' going from 2 to 50 that is 15 elements long and 
    #print it
    lin=np.linspace(2,50,15)
    print(lin)
    #14) Create a 4x3 array of zeros, then add a length 3 array of ones 
    #to each row and print it
    z=np.zeros((4,3))
    z+=np.ones(3)
    print(z)
    #15) Make a 3x3 array 'C' by making an array of length 9 from 1 to 9 and
    #reshaping it to 3x3 and print it
    C0=np.arange(1,10,1)
    C=np.reshape(C0,(3,3))
    print(C)
    #16) Print the eigenvalues and eigenvectors of C
    print(np.linalg.eig(C))
    #17) Make a 3x3 array 'A' like 'C', but ranging from 2 to 10, and print the
    #matrix with the even rows and odd columns indexed
    A0=np.arange(2,11,1)
    A=np.reshape(A0,(3,3))
    print(A[::2,1::2])
    #18) Print the Hadamard (element-wise) and matrix products of A and C
    print(A*C,A@C)
    #19) Generate a 2x3 matrix with elements randomly drawn from the standard
    #normal distribution, then append a length 3 vector with random integers from
    #1 to 5 as the third row
    rand=np.random.randn(2,3)
    rand=np.append(rand,[np.random.randint(1,5,3)],axis=0)
    print(rand)
    #20) Generate a random array of integers of length 10 ranging from 1 to 100,
    #then mask out any even numbers, leaving only the odd remaining, and print
    #the array pre- and post-mask
    rand2=np.random.randint(1,100,10)
    randmask=rand2[np.where(rand2%2==1)]
    print(rand2,randmask)