import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
from scipy.linalg import dft

def rect(x):
    r = np.empty(len(x))
    for i in np.arange(0, len(x), 1):
        if abs(x[i]) > 1:
            r[i] = 0
        else:
            r[i] = 0.5
    return r

# phase = np.pi / 4
# t = np.linspace(0, 10, num=2000, endpoint=False)
# y = rect(t)
# Y = scipy.fftpack.fftshift(scipy.fftpack.fft(y))
# f = scipy.fftpack.fftshift(scipy.fftpack.fftfreq(len(t)))
#
# p = np.angle(Y)
# p[np.abs(Y) < 1] = 0
# plt.plot(f, p)
# plt.show()

# 1st one
n = 400
x = np.linspace(-10,10,n)
s = (np.amax(x)-np.amin(x))/(n-1)
uniform=rect(x)
shift = lambda x:np.append(x[int(n/2):],x[:int(n/2)])
r_shift=shift(uniform)
F_shift=np.dot(dft(n),r_shift)
F = shift(F_shift)*s
nu = x/(n*s**2)
plt.figure()
plt.xlabel('x')
plt.ylabel('F')
plt.plot(nu,abs(F))
plt.grid()
plt.savefig('FT_RECT_1_3.jpg')
plt.show()