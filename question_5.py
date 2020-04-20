import numpy as np
import matplotlib.pyplot as plt

#define f(t,x,x')
def f(t,x,xp):
	g=10
	return -g
#define partial derivative of f(t,x,x') w.r.t x
def fx(t,x,xp):
	return 0
#define partial derivative of f(t,x,x') w.r.t x'
def fxp(t,x,xp):
	return 0
#define exact solution
def x_true(t):
	g=10
	tf=10
	return -g*t**2/2+g*tf*t/2


ti,tf = 0,10 #end points
alpha,beta = 0,0 #boundary condition
N = 50 #number of subinterval
M_max = 10 #maximum number of iterations
TOL = 1e-4 #Tolerance

h = (tf-ti)/N #stepsize
TK = (beta-alpha)/(tf-ti) #initial guess to x'(t)
w1 = np.zeros(N+1)
w2 = np.zeros(N+1)
t = np.linspace(ti,tf,N+1)
k = 1

def shoting(h,w1,w2,u1,u2,t,N):	
	for i in range(N):
		k11 = h*w2[i]
		k21 = h*f(t[i],w1[i],w2[i])
		
		k12 = h*(w2[i]+k21/2)
		k22 = h*f(t[i]+h/2, w1[i]+k11/2, w2[i]+k21/2)
		
		k13 = h*(w2[i]+k22/2)
		k23 = h*f(t[i]+h/2, w1[i]+k12/2, w2[i]+k22/2)
		
		k14 = h*(w2[i]+k23)
		k24 = h*f(t[i]+h, w1[i]+k13, w2[i]+k23)
		
		w1[i+1] = w1[i] + (k11+2*k12+2*k13+k14)/6
		w2[i+1] = w2[i] + (k21+2*k22+2*k23+k24)/6
		
		
		
		l11 = h*u2
		l12 = h*(fx(t[i],w1[i],w2[i])*u1 + fxp(t[i], w1[i], w2[i])*u2)
		
		l21 = h*(u2+l12/2)
		l22 = h*(fx(t[i]+h/2,w1[i],w2[i])*(u1+l11/2) + fxp(t[i]+h/2, w1[i], w2[i])*(u2+l12/2))
		
		l31 = h*(u2 + l22/2)
		l32 = h*(fx(t[i]+h/2,w1[i],w2[i])*(u1+l21/2) + fxp(t[i]+h/2, w1[i], w2[i])*(u2+l22/2))
		
		l41 = h*(u2+l32)
		l42 = h*(fx(t[i]+h,w1[i],w2[i])*(u1+l31) + fxp(t[i]+h, w1[i], w2[i])*(u2+l22))
		
		u1 = u1 + (l11+2*l21+2*l31+l41)/6
		u2 = u2 +(l12+2*l22+2*l32+l42)/6
	return(w1,u1)

while(k<=M_max):
	w1[0],w2[0] = np.array([alpha,TK])
	u1 = 0
	u2 = 1
	w1,u1 = shoting(h,w1,w2,u1,u2,t,N)
	if(abs(w1[N]-beta)<=TOL):
		print("The correct initial value of x'(t) is ",TK)
		break
	plt.plot(t,w1,'m')
	TK = TK-(w1[N]-beta)/u1
	k = k+1
	
plt.plot(t,w1,'b.',label ='Numerical solution')
plt.plot(t,x_true(t),'m',label ='Analytic solution')


#for plotting candidates solution
ini_guess = np.linspace(30,60,6)
for i in range(6):
	w1[0],w2[0] = np.array([alpha,ini_guess[i]])
	u1 = 0
	u2 = 1
	w1,u1 = shoting(h,w1,w2,u1,u2,t,N)
	plt.plot(t,w1,'g',label = 'candidate solutions')
	
plt.xlabel('t')
plt.ylabel('x(t)')
plt.legend()
plt.xlim(0,10)	
plt.ylim(0,200)
plt.show()	
