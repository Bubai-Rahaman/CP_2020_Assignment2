import numpy as np
from matplotlib import pyplot as plt

a = 1
b = 2
h = 0.1
n = int((b-a)/h)
ini_val = 1	#From given initial condition

#def function
def f(y,t):
	return(y/t - (y/t)**2)

#Exact soution
def y_true(t):
	return(t/(1+np.log(t)))

#Euler method
def Euler(ini_val,a,b,h):
	
	n = int((b-a)/h)
	w = np.zeros(n+1)
	t = np.linspace(a,b,n+1)
	
	w[0] = ini_val 		 #initial condition
	
	for i in range(n):
		k = h*f(w[i],t[i])
		w[i+1] = w[i] + k
	return(w,t)

#Data for exact solution
t1 = np.linspace(1,2,100)
y_exact = y_true(t1)

#solution by Euler's method 
y_Euler,t = Euler(ini_val, a, b, h)


#Error
t = np.linspace(1,2,11)
y = y_true(t)
abs_error = np.zeros(n+1)
rel_error = np.zeros(n+1)

for i in range(n+1):
	abs_error[i] = np.abs(y[i]-y_Euler[i])
	rel_error[i] = np.abs(y[i]-y_Euler[i])/y[i] 

fig=plt.figure()
ax1=fig.add_subplot(121)
ax2=fig.add_subplot(122)

ax2.plot(t,rel_error,'r^',label = 'Relative error')
ax2.plot(t,abs_error,'g*',label = 'Absolute error')
ax2.set_xlabel('$t$')
ax2.set_ylabel('$Error$')
ax2.legend()

ax1.plot(t,y_Euler,'b.',label = "Euler's solution")
ax1.plot(t1, y_exact,'k-',label = "Exact solution")
ax1.set_xlabel('$t$')
ax1.set_ylabel('$y(t)$')
ax1.legend()

plt.show()
