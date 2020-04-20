import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp as bvp

#1st equation
def fun1(x,y):
	return np.vstack((y[1], -np.exp(-2*y[0])))

def bc1(ya,yb):
	return np.array([ya[0], yb[0]-np.log(2)])
	
def y1_true(x):
	return np.log(x)
	
#2nd equation
def fun2(x,y):
	return np.vstack((y[1],y[1]*np.cos(x)-y[0]*np.log(y[0])))

def bc2(ya,yb):
	return np.array([ya[0]-1,yb[0]-np.exp(1)])
	
def y2_true(x):
	return np.exp(np.sin(x))
		
#3rd equation
def fun3(x,y):
	return np.vstack((y[1], -(2*(y[1])**3+(y[0])**2*y[1])/np.cos(x)))

def bc3(ya,yb):
	return np.array([ya[0]-2**(-1/4), yb[0]-12**(1/4)/2])
	
def y3_true(x):
	return np.sqrt(np.sin(x))

#4th equation
def fun4(x,y):
	return np.vstack((y[1], 1/2-(y[1])**2/2-y[0]*np.sin(x)/2))

def bc4(ya,yb):
	return np.array([ya[0]-2, yb[0]-2])

def y4_true(x):
	return 2+np.sin(x)

#1st differential equation
x = np.linspace(1,2,10)
y = np.zeros((2,x.size))
res = bvp(fun1, bc1, x, y)
x_plot1 = np.linspace(1,2,100)
y_plot1 = res.sol(x_plot1)[0]

#2nd differential equation
x =np.linspace(0,np.pi/2,5)
y = np.zeros((2,x.size))
y[0] = 1
res = bvp(fun2,bc2, x, y)
x_plot2 = np.linspace(0,np.pi/2,100)
y_plot2 = res.sol(x_plot2)[0]

#3rd differential equation
x = np.linspace(np.pi/4,np.pi/3,5)
y = np.zeros((2,x.size))
res = bvp(fun3, bc3, x, y)
x_plot3 = np.linspace(np.pi/4,np.pi/3,100)
y_plot3 = res.sol(x_plot3)[0]

#4th differential equation
x = np.linspace(0,np.pi,5)
y = np.zeros((2,x.size))
res = bvp(fun4, bc4, x, y)
x_plot4 = np.linspace(0,np.pi,100)
y_plot4 = res.sol(x_plot4)[0]

fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

#plot#1
ax1.plot(x_plot1,y_plot1,'g.',label = 'Numerical solution')
ax1.plot(x_plot1,y1_true(x_plot1),'r', label = 'Analytic solution')
ax1.legend()
ax1.set_xlabel('x')
ax1.set_ylabel('y(x)')
ax1.set_title(r"$1st problem$")

#plot#2
ax2.plot(x_plot2,y_plot2,'g.',label = 'Numerical solution')
ax2.plot(x_plot2,y2_true(x_plot2),'r', label = 'Analytic solution')
ax2.legend()
ax2.set_xlabel('x')
ax2.set_ylabel('y(x)')
ax2.set_title(r"$2nd problem$")

#plot#3
ax3.plot(x_plot3,y_plot3,'g.',label = 'Numerical solution')
ax3.plot(x_plot3,y3_true(x_plot3),'r', label = 'Analytic solution')
ax3.legend()
ax3.set_xlabel('x')
ax3.set_ylabel('y(x)')
ax3.set_title(r"$3rd problem$")

#plot#4
ax4.plot(x_plot4,y_plot4,'g.',label = 'Numerical solution')
ax4.plot(x_plot4,y4_true(x_plot4),'r', label = 'Analytic solution')
ax4.legend()
ax4.set_xlabel('x')
ax4.set_ylabel('y(x)')
ax4.set_title(r"$4th problem$")

plt.show()
