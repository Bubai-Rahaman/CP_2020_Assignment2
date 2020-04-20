import numpy as np
import matplotlib.pyplot as plt

def p(t):
	return 0

def q(t):
	return 0

def r(t):
	g = 10
	return -10
def x_true(t):
	g=10
	tf=10
	return -g*t**2/2+g*tf*t/2



ti,tf = 0,10 #endpoints
alpha,beta = 0,0 #boundary condition

N = 20
h = (tf-ti)/N #step size
t = np.linspace(ti,tf,N+1)
w = np.zeros(N+1)
v = np.zeros(N+1)
M = 20
A = np.zeros((N+1,N+1))
b = np.zeros(N+1)
TOL = 1e-4 #tolerance
w[0],w[N] = 0,0

#Assigning value to A matrix
A[N,N] = 2+h**2*q(t[N])
for i in range(N):
	A[i,i] = 2+h**2*q(t[i])
	A[i,i+1] = -1+h*p(t[i])/2
	A[i+1,i] = -1-h*p(t[i+1])

#assigning value to b vector
b[0] = -h**2*r(t[0])+(1+h*p(t[0])/2)*alpha
b[N] = -h**2*r(t[N])+(1-h*p(t[N])/2)*beta
for i in range(1,N):
	b[i] = -h**2*r(t[i])

k = 0
while True:
	for i in range(1,N):
		z=0
		for j in range(1,N):
			z = z+w[j]*A[i,j]
		w[i] = v[i] + (b[i]-z)/A[i,i]
	if (np.sqrt(np.dot(w-v,w-v))<TOL):
		break
	plt.plot(t,w,'g',label = 'candidate solutions')
	v = w.copy()

t_plot = np.linspace(ti,tf,100)
plt.plot(t,w,'ro', label = 'Numerical solution',)
plt.plot(t_plot,x_true(t_plot),'b',label='exact solution')
plt.legend()
plt.xlabel('t')
plt.ylabel('y(t)')
plt.show()	
