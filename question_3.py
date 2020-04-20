import numpy as np
from matplotlib import pyplot as plt

def f1(w1,w2,x):
	return w2

def f2(w1,w2,x):
	return x*(np.exp(x)-1)+2*w2-w1

def y_true(x):
	return -2-x+np.exp(x)*(12-6*x+x**3)/6

a,b,h = 0,1,0.01
n = int((b-a)/h)

w1 = np.zeros(n+1)
w2 = np.zeros(n+1)

w1[0]=0
w2[0]=0
x = np.linspace(0,1,n+1)
for i in range(n):
	k11 = h*f1(w1[i],w2[i],x[i])
	k21 = h*f2(w1[i],w2[i],x[i])
	
	
	k12 = h*f1(w1[i]+k11/2,w2[i]+k21/2,x[i]+h/2.)
	k22 = h*f2(w1[i]+k11/2,w2[i]+k21/2,x[i]+h/2.)
	
	
	k13 = h*f1(w1[i]+k12/2,w2[i]+k22/2,x[i]+h/2.)
	k23 = h*f2(w1[i]+k12/2,w2[i]+k22/2,x[i]+h/2.)
	
	k14 = h*f1(w1[i]+k13,w2[i]+k23,x[i]+h)
	k24 = h*f2(w1[i]+k13,w2[i]+k23,x[i]+h)
	
	w1[i+1] = w1[i]+(k11+2*k12+2*k13+k14)/6
	w2[i+1] = w2[i] + (k21+2*k22+2*k23+k24)/6
	
plt.plot(x,w1,'m*',label = 'Numerical solution')
plt.plot(x,y_true(x),'g',label = 'Analytic solution')
plt.legend()
plt.xlabel('x')
plt.ylabel('y(x)')
plt.yscale('log')
plt.show()
