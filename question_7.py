import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp as ivp

#1st differential equation
def f1(t,y):
	return t*np.exp(3*t)-2*y

def y1_true(t):
	return np.exp(-2*t)*(1-np.exp(5*t)+5*t*np.exp(5*t))/25

#2nd differentia equation
def f2(t,y):
	return 1-(t-y)**2

def y2_true(t):
	return (1-5*t+t**2)/(-3+t)

#3rd differential equation
def f3(t,y):
	return 1+y/t

def y3_true(t):
	return 2*t +t*np.log(t)

#4th differential equation
def f4(t,y):
	return np.cos(2*t)+np.sin(3*t)
	
def y4_true(t):
	return (8-2*np.cos(3*t)+3*np.sin(2*t))/6

#solution of 1st eqn
sol1 = ivp(f1,[0,1],[0],dense_output =True)	
t1 = np.linspace(0,1,100)
y1 = sol1.sol(t1)

#solution of 2nd equation]
sol2 = ivp(f2,[2,3],[1],dense_output=True)
t2 = np.linspace(2,3,100)
y2 = sol2.sol(t2)

#solution of 3rd equation
sol3 = ivp(f3,[1,2],[2],dense_output=True)
t3 = np.linspace(1,2,100)
y3 = sol3.sol(t3)

#solution of 4th equation
sol4 = ivp(f4,[0,1],[1],dense_output=True)
t4 = np.linspace(0,1,100)
y4 = sol4.sol(t4)



fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

ax1.plot(t1,y1.T,'r.',label = 'Numerical solution')
ax1.plot(t1,y1_true(t1),'k-',label = 'Analytic solution')
ax1.legend()
ax1.set_xlabel('$t$')
ax1.set_ylabel('$y(t)$')
ax1.set_title(r"y'=t*e^3t-ty")

ax2.plot(t2,y2.T,'r.',label = 'Numerical solution')
ax2.plot(t2,y2_true(t2),'k-',label = 'Analytic solution')
ax2.legend()
ax2.set_xlabel('$t$')
ax2.set_ylabel('$y(t)$')
ax2.set_title(r"y'=1-(t-y)^2")

ax3.plot(t3,y3.T,'r.',label = 'Numerical solution')
ax3.plot(t3,y3_true(t3),'k-',label = 'Analytic solution')
ax3.legend()
ax3.set_xlabel('$t$')
ax3.set_ylabel('$y(t)$')
ax3.set_title(r"y'=1+y/t")

ax4.plot(t4,y4.T,'r.',label = 'Numerical solution')
ax4.plot(t4,y4_true(t4),'k-',label = 'Analytic solution')
ax4.legend()
ax4.set_xlabel('$t$')
ax4.set_ylabel('$y(t)$')
ax4.set_title(r"y'=cos(2t)+sin(3t)")

plt.show()
