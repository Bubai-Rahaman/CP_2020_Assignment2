import numpy as np
import matplotlib.pyplot as plt

#define dy/dt
def f(t,y):
	return (y**2+y)/t

#exact solution
def y_true(t):
	return 2*t/(1-2*t)
	
ti,tf =1,3 	#range of t value
ini_val = -2	#initial condition

h_min = 0.01 	#minimum step size
N_max = int((tf-ti)/h_min)	#maximum number of steps

y = np.zeros(N_max+1) 		#array to store the solution
t = np.zeros(N_max+1)		#array for the mesh points

t[0] = ti
y[0] = ini_val
Tol = 1e-4				#tolarence
i = 0
j = 0
h = h_min				#initial stepsize				

while(t[j]<tf):
	
	#with 2*h step size
	k1 = 2*h*f(t[j],y[j])
	k2 = 2*h*f(t[j]+h,y[j]+k1/2)
	k3 = 2*h*f(t[j]+h,y[j]+k2/2)
	k4 = 2*h*f(t[j]+2*h,y[j]+k3)
	w = y[j]+(k1+2*k2+2*k3+k4)/6
		
	#with step size h
	for i in range(2):
		k1 = h*f(t[i+j],y[i+j])
		k2 = h*f(t[i+j]+h/2,y[i+j]+k1/2)
		k3 = h*f(t[j+i]+h/2,y[i+j]+k2/2)
		k4 = h*f(t[j+i]+ h,y[i+j]+k3)
		y[i+j+1] = y[i+j]+(k1+2*k2+2*k3+k4)/6
		t[i+j+1] = t[i+j]+h
	
	delta = abs(y[j+2]-w)		#dfferene between the two solution with step size h and 2h
	
	for i in range(2):
		y[i+j+1] += delta/15	#adding correction term to the solution
	
	k = h*(Tol/delta)**0.2
	
	if (t[j+2]+2*h>tf):
		h = (tf-t[j+2])/2
	elif (k<h):
		h = h_min
	else:
		h = k
	
	j = j+2

y.resize((j+1))		#reducing size of the array
t.resize((j+1))

t_plot = np.linspace(1,3,100)
#plot
plt.plot(t,y,'m*',label = 'Numerical solution')
plt.plot(t_plot,y_true(t_plot),'g',label = 'Exact solution')
plt.legend()
plt.xlabel('t')
plt.ylabel('y(t)')
plt.grid()
plt.show()

