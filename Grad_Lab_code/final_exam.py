##IMGS-609 Python Grad Lab Final Exam##
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import cv2
from PIL import Image
from scipy.linalg import dft

##Basics (Answer 3)##

#1) Create a third-degree polynomial function (i.e. ax^3+...) and solve it
#for x=5, a=4, b=3, c=2, and d=1, printing the result

#2) Create a list that consists of first a floating number 2, the boolean
#False, and finally another list containing the strings 'Math' and 'Python'
#in that order, then append the integer 1 and print the result
list_memo = [2. , False, ['Math', 'Python']]
list_memo.append(1)
print(list_memo)

#3) Create a tuple with the same elements as the list, then print the tuple
#as well as the even elements of the tuple via index slicing
tuple_memo = (2. , False, ['Math', 'Python'], 1)
print(tuple_memo, tuple_memo[0::2])


#4) Make a dictionary containing key 'Time' and value 3 (either float or integer), 
#key 'Exam' and value 'final', and key 'End' and value True, then call any 
#value by its key and print it
dict_memo = {'Time': 3, 'Exam': 'final', 'End': True}
print(dict_memo['Time'])


##Numpy (Answer 4)##

#5) Create a numpy array 'A' where the first row is [1,5,6], the second is [2,8,4],
#and the third is [3,9,7], and print the result
A = np.array([[1, 5, 6],[2, 8, 4],[3, 9, 7]])
print(A)

#6) Print the eigenvalues and eigenvectors of A, then create a new array B 
#by appending the eigenvalues to A as the fourth row and print B
W, V = np.linalg.eig(A)
print(W)
print(V)
B = np.row_stack((A,W))
print(B)

#7) Print the mean and sum of B; also, print the amount of computer memory
#taken by B in bytes
mean_B = np.mean(B)
print(mean_B)
sum_B = np.sum(B)
print(sum_B)
print(B.nbytes)

#8) Create a 3x3 matrix C by reshaping a length 9 array that goes from 3 to 
#11, then print both the Hadamard/element-wise and the matrix/dot product of 
#A with C
C0=np.arange(3,12,1)
C=np.reshape(C0,(3,3))
print(A*C)
print(A@C)


#9) Generate a random length 20 array with numbers drawn from the standard
#normal distribution; print it, then mask out all numbers less than zero
#and print the resulting array of positive numbers


##Numpy, Matplotlib, and Scipy (Answer 4)##

#10) Plot y=cos(x)^2 with a red dash-dot line of alpha 0.25 and scatter plot 
#y=sin(x)^2 with green triangles of size 20 on the same graph from x=-5 to 5
#with at least 100 points, with labeled x and y axes, a legend, and titled 
#"Squared Trig. Functions"
x=np.arange(-5,5,0.05)
y1 = np.cos(x)**2
y2 = np.sin(x)**2
plt.figure()
plt.title('Squared Trig. Plot')
plt.plot(x,y1,'r--',alpha=0.25, label='cos')
plt.plot(x,y2,'g^',ms=20, label='sin')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

#11) Now create stacked subplots with a shared x-axis; on the top, use the 
#squared cosine function from before (including alpha and line style) but 
#with a logarithmically scaled y-axis; on the bottom, plot log10(cos(x)^2) 
#with a regularly scaled y-axis; label and title the plot appropriately, 
#and keep x limited from -3 to 3
x=np.arange(-3,3,0.01)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.set_yscale('log')
ax1.set_title('Logarithmic Comparison')
ax1.plot(x, np.cos(x)**2,'r--', alpha=0.25, label='cos with y log scaled')
ax2.plot(x, np.log10(np.cos(x)**2), label='log10(cos(x)^2) with regularly scaled y')
ax1.legend()
ax2.legend()
plt.show()

#12) Take your third degree polynomial function from earlier and fit points
#(3,0), (4,1), (7,6), and (8,0) to a curve; plot the line and scatter plot 
#the points on the function; label the axes and title the plot with the
#found coefficients
func=lambda x,a,b,c,d: a*x**3+b*x**2+c*x+d
pts=np.array([[3,0],[4,1],[7,6],[8,0]])
fit_curve,_=opt.curve_fit(func,pts[:,0],pts[:,1])
p_range=np.arange(0,10,0.01)
plt.figure()
plt.title('a='+str(fit_curve[0])+',b='+str(fit_curve[1])+',c='+str(fit_curve[2])+',d='+str(fit_curve[3]), fontsize=8)
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(pts[:,0],pts[:,1])
plt.plot(p_range,func(p_range,fit_curve[0],fit_curve[1],fit_curve[2],fit_curve[3]))
# print(opt.root(func,[0,5],args=(fit_curve[0],fit_curve[1],fit_curve[2],fit_curve[3])).x)
plt.show()

#13) Use this plot again, but create an interpolation function with the 4
#initial points, then interpolate (linear) at least 50 points within the x 
#range 3 to 8 and scatter plot the interpolated points on the graph (separately
#from the 4 main points) using markers of size 5; include a legend, and this time
#title the plot 'Interpolation Comparison'

#14) Use the plot one more time, and find the x values of the minimum and 
#maximum y values in the range from x=3 to x=8, then separately scatter plot
#those two points with marker size 100; include the x and y coordinates of
#the minimum and maximum points in the title


#15) Create a function that is equal to 1 for all x, then plot the Fourier
#Transform of the function against the frequency; label the axes and title
#the plot accordingly (HINT: for this function in particular, shifting before
#the transform will not be necessary, but will be afterwards)
def func(x):
    r = np.ones(len(x))
    return r

n = 400
x = np.linspace(-5,5,n)
s = (np.amax(x)-np.amin(x))/(n-1)
uniform=func(x)
shift = lambda x:np.append(x[int(n/2):],x[:int(n/2)])
r_shift=shift(uniform)
F_shift=np.dot(dft(n),r_shift)
F = shift(F_shift)*s
nu = x/(n*s**2)
plt.figure()
plt.title('Fourier Transform for uniformly distributed function')
plt.xlabel('x')
plt.ylabel('F')
plt.plot(nu,F)
plt.show()


##Image Processing (Answer 4)##
#(Image courtesy of https://www.pumpkin.care/blog/cat-age-chart/)

#16) Load in the attached cat image using both OpenCV and PIL; show them 
#side-by-side: Left, from OpenCV BEFORE RGB conversion, Center: from PIL,
#Right: from OpenCV AFTER RGB conversion
cat_cv=cv2.imread('cat.jpg')
cat_pil=Image.open('cat.jpg')
cat_conversion=cv2.cvtColor(cat_cv,cv2.COLOR_BGR2RGB)
fig,ax=plt.subplots(1,3)
ax[0].set_title('cv_imread')
ax[0].imshow(cat_cv,cmap='gray')
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].set_title('Image read')
ax[1].imshow(cat_pil,cmap='gray')
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[2].set_title('cv_cvtcolor')
ax[2].imshow(cat_conversion,cmap='gray')
ax[2].set_xticks([])
ax[2].set_yticks([])
plt.show()

#17) Use a 25x25 box (average) filter (that is, all 1s divided by 25^2) to 
#blur the RGB cat image and show the result

#18) Double the size of the image, rotate the image 80 degrees clockwise,
#and show the result
resize_cat=cat_pil.resize((2*int(np.shape(cat_pil)[1]),2*int(np.shape(cat_pil)[0])),resample=Image.BILINEAR)
# print(resize_cat)
rotate_cat=resize_cat.rotate(80)
plt.figure()
plt.title('Cat double size and rotate 80')
plt.imshow(rotate_cat)
plt.xticks([])
plt.yticks([])
plt.show()

#19) Gamma correct the cat image with gamma values of 0.25 and 4, and show
#the results side by side
lcs=lambda img: 255 * (img - np.amin(img)) / (np.amax(img) - np.amin(img))
gamma_correction= lambda img, g:lcs(img**g)
fig,ax=plt.subplots(1,2)
ax[0].set_title('Cat with Gamma=0.25')
ax[0].imshow(gamma_correction(cat_conversion.astype('float64'), 0.25).astype('uint8'))
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].set_title('Cat with Gamma=4')
ax[1].imshow(gamma_correction(cat_conversion.astype('float64'), 4).astype('uint8'))
ax[1].set_xticks([])
ax[1].set_yticks([])
plt.show()

#20) Use the "flatten" attribute of the cat image and plot the histogram of
#the full cat image with 255 bins, with the graph labeled accordingly
data = np.array(cat_pil)
flattened = data.flatten()
plt.hist(flattened, bins=256)
plt.xlabel('grayscale value')
plt.ylabel('pixel amount')
plt.title('Histogram')
plt.show()