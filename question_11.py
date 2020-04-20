import numpy as np
import matplotlib.pyplot as plt

def f1(t,u1,u2,u3):
	return u1 + 2*u2 -2*u3 + np.exp(-t)

def f2(t,u1,u2,u3):
	return u2+u3-2*np.exp(-t)

def f3(t,u1,u2,u3):
	return u1 +2*u2 +np.exp(-t)

def y1(t):
	return -3*np.exp(-t)*(-2*np.exp(t)*np.cos(t)+np.exp(t)*np.sin(t)+1)

ti,tf = 0,1
h = 0.01
N = int((tf-ti)/h)
t = np.linspace(ti,tf,N+1)
u1 = np.zeros(N+1)
u2 = np.zeros(N+1)
u3 = np.zeros(N+1)

u1[0],u2[0],u3[0] = 3,-1,1

for i in range(N):
	k11 = h*f1(t[i],u1[i],u2[i],u3[i])
	k21 = h*f2(t[i],u1[i],u2[i],u3[i])
	k31 = h*f3(t[i],u1[i],u2[i],u3[i])
	
	k12 = h*f1(t[i]+h/2,u1[i]+k11/2,u2[i]+k21/2,u3[i]+k31/2)
	k22 = h*f2(t[i]+h/2,u1[i]+k11/2,u2[i]+k21/2,u3[i]+k31/2)
	k32 = h*f3(t[i]+h/2,u1[i]+k11/2,u2[i]+k21/2,u3[i]+k31/2)
	
	k13 = h*f1(t[i]+h/2,u1[i]+k12/2,u2[i]+k22/2,u3[i]+k32/2)
	k23 = h*f2(t[i]+h/2,u1[i]+k12/2,u2[i]+k22/2,u3[i]+k32/2)
	k33 = h*f3(t[i]+h/2,u1[i]+k12/2,u2[i]+k22/2,u3[i]+k32/2)
	
	k14 = h*f1(t[i]+h,u1[i]+k13,u2[i]+k23,u3[i]+k33)
	k24 = h*f2(t[i]+h,u1[i]+k13,u2[i]+k23,u3[i]+k33)
	k34 = h*f3(t[i]+h,u1[i]+k13,u2[i]+k23,u3[i]+k33)
	
	u1[i+1] = u1[i] + (k11+2*(k12+k13)+k14)/6
	u2[i+1] = u2[i] + (k21+2*(k22+k23)+k24)/6
	u3[i+1] = u3[i] + (k31+2*(k32+k33)+k34)/6

plt.plot(t,u1,'g.-', label = 'u1')
plt.plot(t,u2,'m*-', label = 'u2')
plt.plot(t,u3,'b^-', label = 'u3')
plt.legend()
plt.xlabel('t')
plt.ylabel('u1(t),u2(t),u3(t)')
plt.show()
