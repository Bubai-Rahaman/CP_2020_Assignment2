from scipy.optimize import root
import numpy as np
import matplotlib.pyplot as plt

#1st differential equation
def euler1(ini_val,a,b):
	n = 20
	h = (b-a)/n
	w = np.zeros(n+1)
	x = np.linspace(a,b,n+1)
	w[0] = ini_val
	for i in range(n):
		w[i+1] = w[i]/((1+9*h)**(i+1))
	return(x,w)

#exact solution
def y1(x):
	return np.exp(-9*x+1)
	

a = 0
b = 1
ini_val = np.e
x,y = euler1(ini_val,a,b)

#2ND differential equation
def f(w):
	return w-h*(-20*(w-t)**2+2*t)-w0

alpha = 0
beta = 1
ini_cond = 1/3

n = 100
h = (beta-alpha)/n
w = np.zeros(n+1)
x2 = np.linspace(alpha,beta,n+1)
w[0] = ini_cond	
for i in range(n):
	w0 = w[i]
	t = x2[i]
	sol = root(f,x0=w[i])
	w[i+1] = sol.x
	


fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.plot(x,y,'g.',label = "Backward Euler solution")
ax1.plot(x,y1(x),'b',label = 'Exact solution')
ax1.set_xlabel('$x$')
ax1.set_ylabel('$y(x)$')
ax1.legend()


ax2.plot(x2,w,'g*-',label = "Backward Euler solution")
ax2.set_xlabel('$x$')
ax2.set_ylabel('$y(x)$')
ax2.legend()

ax1.set_title(r"$y'(x)=-9y(x) $")
ax2.set_title(r"$y'(x)= -20(y(x)-x)^2+2x$")
plt.show()
