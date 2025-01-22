import numpy as np
import matplotlib.pyplot as mpl
import scipy.sparse

# Parameters
n = 7  
h = 1 / (n + 1) 
g_0 = 0  
g_1 = 2  
p = 0  
q = 4  
x = np.linspace(0, 1, n + 2)  

# Define function f(x)
def f(x):
    return 4 * x

# Construct finite difference matrix A
main_diag = 2 + q * h**2  # main diagonal
sub_diag = -1  # sub-diagonal
super_diag = -1  # super-diagonal

diagonals = [sub_diag * np.ones(n - 1), main_diag * np.ones(n), super_diag * np.ones(n - 1)]
A = scipy.sparse.diags(diagonals, offsets=[-1, 0, 1]).toarray()

# Construct vector B
b = (h**2) * f(x[1:-1])  
b[0] += g_0  
b[-1] += g_1  

# Solve the linear system
from scipy.linalg import solve
U = solve(A, b)

# Add boundary values to U
U_full = np.concatenate(([g_0], U, [g_1]))

# Shooting Method Definitions
def z_p(x, y):
    return 4 * y - 4 * x

def y_p(z):
    return z

y0 = 0
y1 = 2
n_shoot = 8
h_shoot = 1 / n_shoot

# Initial guesses for z0
z0_g1 = 0.5
z0_g2 = 1.0

def solve_ivp(z0):  # solve IVP for given z0
    yvalues = [y0]
    zvalues = [z0]
    xvalues = np.linspace(0, 1, n_shoot + 1)
    for i in range(n_shoot):
        z_next = zvalues[i] + h_shoot * z_p(xvalues[i], yvalues[i])
        y_next = yvalues[i] + h_shoot * y_p(zvalues[i])
        zvalues.append(z_next)
        yvalues.append(y_next)
    return yvalues[-1]

# Secant Method for Shooting
tol = 1e-6
max_iter = 100
f1 = solve_ivp(z0_g1) - y1
f2 = solve_ivp(z0_g2) - y1

for _ in range(max_iter):
    z_new = z0_g2 - f2 * (z0_g2 - z0_g1) / (f2 - f1)
    f_new = solve_ivp(z_new) - y1
    if abs(f_new) < tol:
        z0_correct = z_new
        break
    z0_g1, z0_g2 = z0_g2, z_new
    f1, f2 = f2, f_new

# Compute Shooting Solution
yvalues = [y0]
zvalues = [z0_correct]
xvalues = np.linspace(0, 1, n_shoot + 1)
for i in range(n_shoot):
    z_next = zvalues[i] + h_shoot * z_p(xvalues[i], yvalues[i])
    y_next = yvalues[i] + h_shoot * y_p(zvalues[i])
    zvalues.append(z_next)
    yvalues.append(y_next)

# Plot Solutions
mpl.plot(x, U_full, marker='o', linestyle='-', color='b', label='Finite Difference')
mpl.plot(xvalues, yvalues, marker='o', linestyle='-', color='g', label='Shooting Method')
mpl.xlabel('x')
mpl.ylabel('U')
mpl.ylim(0, 2)
mpl.title(f"Solution of BVP (Finite Difference & Shooting, n={n_shoot})", fontsize=18)
mpl.legend()
mpl.grid(True)
mpl.show()
