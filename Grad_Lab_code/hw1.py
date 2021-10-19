from __future__ import division
import numpy as np

#1) Print your name
print('Cheng Han')

#2) Use lambda keyword to write a function 'func' that squares an input
#and adds 5 to it, then calculate it with x=3 and print the result
func = lambda x: x**2 + 5
print(func(3))

#3) Use def keyword to create a function 'newfunc' that returns the input
#squared plus 5 if less than 5 (can use 'func' if desired) and multiplies
#it by 12 otherwise, then calculate with x=3 and x=7 and print the results
def new_func(x):
    if x < 5:
        return func(x)
    else:
        return x*12
print(new_func(3))
print(new_func(7))

#4) Make a list 'foo' with the first element being the string 'Grad Lab',
#the second being the boolean True, the third being a list consisting of
#integer 2 and float 4, then print it
foo = ['Grad Lab', True, [int(2), float(4)]]
print(foo)

#5) Append the float number 1 to the end of foo, then delete the boolean
#in index 1, then print the result
foo.append(float(1))
for i in range(len(foo)):
    if foo[i]:
        foo.pop(i)
        break
print(foo)

#6) Make a tuple 'too' using the same initial elements of 'foo' along with
#the floating 1 at the end and print it
too = ('Grad Lab', True, [int(2), float(4)], float(1))
print(too)

#7) Print the second and third elements of both 'foo' and 'too' using
#index slicing
print(foo[1:3])
print(too[1:3])

#8) Print the odd elements of both using index slicing
print(foo[::2])
print(too[::2])


#9) Create a dictionary 'course' with 'class' as the key for 'Grad Lab',
#'taking' as the key for True, 'times' as the key for the 2 and 4 list, and
#'assignment' as the key for 1, then print any one result by calling the
#corresponding key from the dictionary
course = {
    'class': 'Grad Lab', 'taking': True, 'times': [2, 4], 'assignment': 1
}
print(course['class'])
print(course['taking'])
print(course['times'])
print(course['assignment'])

#10) Use a for loop to calculate 0+1+2+..+100 and print the result
result = 0
for i in range(0, 101, 1):
    result += i
print(result)

#11) Create a Numpy array 'npi' consisting of the numbers 3,5,15,10,11,
#and print both the array and its maximum value
npi = np.array([3, 5, 15, 10, 11])
print(npi)
print(np.max(npi))

#12) Make an array 'ara' going from 2 to 50 in increments of 2 and print it
ara = [x for x in range(2, 51, 2)]
print(ara)

#13) Make an array 'lin' going from 2 to 50 that is 15 elements long and
#print it
lin = np.linspace(2, 50, 15)
print(lin)

#14) Create a 4x3 array of zeros, then add a length 3 array of ones
#to each row and print it

zeros = np.zeros((4, 3))
print(zeros + 1)

#15) Make a 3x3 array 'C' by making an array of length 9 from 1 to 9 and
#reshaping it to 3x3 and print it
C = np.arange(1, 10)
C = C.reshape((3,3))
print(C)

#16) Print the eigenvalues and eigenvectors of C
print('eigenvalues', np.linalg.eig(C)[0])
print('eigenvectors', np.linalg.eig(C)[1])


#17) Make a 3x3 array 'A' like 'C', but ranging from 2 to 10, and print the
#matrix with the even rows and odd columns indexed
A = np.arange(2, 11).reshape((3, 3))
for i in range(len(A)):
    for j in range(len(A[0])):
        if i%2 ==0 and j%2 !=0:
            print(A[i][j])


#18) Print the Hadamard (element-wise) and matrix products of A and C
print('Hadamard', A*C)
print('matrix products', np.dot(A, C))

#19) Generate a 2x3 matrix with elements randomly drawn from the standard
#normal distribution, then append a length 3 vector with random integers from
random_normal = np.random.normal(0, 0.1, (2,3))
print(random_normal)

#20) Generate a random array of integers of length 10 ranging from 1 to 100,
#then mask out any even numbers, leaving only the odd remaining, and print
#the array pre- and post-mask
array_pre_mask = np.random.randint(1, 100, 10)
print(array_pre_mask)
array_post_mask = []
for num in array_pre_mask:
    if num%2 != 0:
        array_post_mask.append(num)
print(array_post_mask)
