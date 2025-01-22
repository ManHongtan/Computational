import numpy as np
import matplotlib.pyplot as mpl
import scipy.sparse

#initialised parameters
n=3
h=1/(n+1)
g_0=0
g_1=2
p=0
q=4
x=np.linspace(0, 1, n + 2)

#define function
def f(x):
    return 4*x

#create diagonal matrix A
diagonal=np.zeros((n,n))
diagonal[n-n,:]=-(1.0+h*p/2)
diagonal[n-2,:]=2+q*h**2
diagonal[n-1,:]=-(1.0-h*p/2)

#transform diagonalised matrix
A=scipy.sparse.spdiags(diagonal,[-1,0,1],n,n,format='csc')
A=A.toarray()
print(A)

#create matrix B
b=np.zeros(n)
b[0]=(1.0+h*p/2)*g_0+(h**2)*f(x[1])
b[1]=(h**2)*f(x[2])
b[2]=(1-h*p/2)*g_1+(h**2)*f(x[3])
print(b)

from scipy.linalg import solve
U=solve(A,b)
print(U)


# Construct the full solution including boundary conditions
U_full = np.concatenate(([g_0], U, [g_1]))


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


# Plot the solution
mpl.plot(x, U_full, marker='o', linestyle='-', color='b')
mpl.plot(xvalues,yvalues,marker='o',linestyle='-', color='g')
mpl.xlabel('x')
mpl.ylabel('U')
mpl.ylim(0,2)
mpl.title(f"Solution of BVP(finite difference,n={n})",fontsize=18)
mpl.legend()
mpl.grid(True)
mpl.show()

