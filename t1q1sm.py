import numpy as np
import matplotlib.pyplot as mpl

#define equations and boundary conditions
def z_p(x,y):
    return 4*y-4*x

def y_p(z):
    return z

y0=0
y1=2
x0=0
n=4
h=1/n

#initial guess for z0
z0_g1=0.5
z0_g2=1.0

def solve_ivp(z0):  #solve ivp for initial slope z0
    yvalues=[y0]
    zvalues=[z0]
    xvalues=np.linspace(0,1,n+1)

    for i in range(n):
        z_next=zvalues[i]+h*z_p(xvalues[i],yvalues[i])
        y_next=yvalues[i]+h*y_p(zvalues[i])
        zvalues.append(z_next)
        yvalues.append(y_next)

    return yvalues[-1]

#secant method
tol=1e-6
max_i=100

f1=solve_ivp(z0_g1)-y1
f2=solve_ivp(z0_g2)-y1

for i in range(max_i):
    z_new=z0_g2-f2*(z0_g2-z0_g1)/(f2-f1)    #update secant
    f_new=solve_ivp(z_new)-y1   #evaluate bvp with new guess
    if abs(f_new)<tol:  #check for convergence
        z0_correct=z_new
        break
    z0_g1,z0_g2=z0_g2,z_new
    f1,f2=f2,f_new

yvalues=[y0]
zvalues=[z0_correct]
xvalues=np.linspace(0,1,n+1)

for i in range(n):
    z_next=zvalues[i]+h*z_p(xvalues[i],yvalues[i])
    y_next=yvalues[i]+h*y_p(zvalues[i])
    zvalues.append(z_next)
    yvalues.append(y_next)

#print results
print(f"Correct initial slope(z0):{z0_correct}")
print("x-values:",xvalues)
print("y-values:",yvalues)

#plot
mpl.figure(figsize=(8,6))
mpl.plot(xvalues,yvalues,marker='o')
mpl.xlabel('x',fontsize=14)
mpl.ylabel('y',fontsize=14)
mpl.title(f"Solution of BVP(shooting method,n={n})",fontsize=18)
mpl.grid(True)
mpl.legend(fontsize=10)
mpl.show()