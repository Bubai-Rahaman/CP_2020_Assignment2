#include<stdio.h>
#include<math.h>

/*
Here f(t,y) is given as (y-t^2+1). Then from lipschitz length is L obtained from the partial derivative of f(t,y) w.r.t y(t) as L = 1.

And |y''(t)|<= 0.5e^2-2. so M = 0.5e^2-2
*/

float f(float y, float t)
{
	return(y-t*t+1);
}

//exact solution
float y_true(float t)
{
	return(pow(t+1,2)-0.5*exp(t));
}

float euler_method(float y[],float ini_val, int n, float h)
{
	int i;
	float k;
	y[0] = ini_val;
	
	for(i=0; i<n+1; i++)
	{
		k = h*f(y[i],i*h);
		y[i+1] = y[i] + k;
	}
	return (y[n]);
}

void main()
{
	float a = 0.0, b= 2.0,h = 0.2;
	int i,n;
	n = (b-a)/h;
	float E[n+1],ini_val,y[n+1],w[n+1];
	float L = 1.0, M = 0.5*exp(2)-2;
	
	//error upper bound
	for(i=0;i<n+1;i++)
	{
		E[i]=h*M*(exp(i*h)-1)/(2*L);
	}
	
	//assiging initial valiue
	ini_val = 0.5;
	euler_method(w,ini_val,n,h);
	printf("ti\t\tActual error\tError bound\n");
	
	for(i=1;i<n+1;i++)
	{
		y[i] = y_true(a+i*h);
		printf("%f\t%f\t%f\n",a+i*h,fabs(y[i]-w[i]),E[i]);
	}
	
}
