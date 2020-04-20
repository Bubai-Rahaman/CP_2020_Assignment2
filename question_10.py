import numpy as np
import matplotlib.pyplot as plt

'''
Changing the argument from "t" to "v=1/(1+t)"
'''

def f(x,v):
	return -1/(x**2*v**2+(1-v)**2)

vi,vf = 1,0
h = -0.01
N = int((vf-vi)/h)

v = np.linspace(vf,vi,N+1)
x = np.zeros(N+1)
x[N] = 1

for i in range(N):
	k1 = h*f(x[N-i],v[N-i])
	k2 = h*f(x[N-i]+k1/2,v[N-i]+h/2)
	k3 = h*f(x[N-i]+k2/2,v[N-i]+h/2)
	k4 = h*f(x[N-i]+k3,v[N-i]+h)
	
	x[N-1-i] = x[N-i]+(k1+2*k2+2*k3+k4)/6

t = 3.5*10**6
v0 = 1/(1+t)

z = np.polyfit(v,x,3)
p = np.poly1d(z)

print("The value of the function at the given point is ",p(v0))

plt.plot(v,x,'g.-',label = 'Numerical solution')
plt.legend()
plt.xlabel('v=1/(1+t)')
plt.ylabel('x')
plt.show()
