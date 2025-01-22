import numpy as np

x0=0
y0=1
n=4
h=1/n

def f(x,y):
    return y*(1-np.exp(2*x))

#euler method
x=np.linspace(0,1,n+1)
y=np.zeros(n+1)
y[0]=y0

for i in range(n):
    y[i+1]=y[i]+h*f(x[i],y[i])

for xi, yi in zip(x, y):
    print(f"x: {xi:.2f}, y: {yi:.6f}")

    


