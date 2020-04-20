import numpy as np
import matplotlib.pyplot as plt

def f1(w1,w2,t):
	return ((t**3)*np.log(t)+2*t*w1-2*w2)/t**2

def f2(w1,w2,t):
	return w1

def y(t):
	return 7*t/4+(t**3)*np.log(t)/2-3*(t**3)/4
	

ti,tf = 1,2
h = 0.001
n = int((tf-ti)/h)

t = np.linspace(ti,tf,n+1)
w1 = np.zeros(n+1)
w2 = np.zeros(n+1)

w1[0],w2[0] = 0,1

for i in range(n):
	k1 = h*f1(w1[i],w2[i],t[i])
	w1[i+1] = w1[i]+k1
	k2 = h*f2(w1[i],w2[i],t[i])
	w2[i+1] = w2[i] + k2
	
plt.plot(t,w2,'b*',label = 'Numerical solution')
plt.plot(t,y(t),'r',label = 'Analytic solution')
plt.legend()
plt.xlabel('t')
plt.ylabel('y(t)')
plt.show()

