import numpy as np
import matplotlib.pyplot as mpl

# Finite Difference Method
# Initialized parameters
n = 1
h = np.pi / 4
g_0 = -0.3
g_1 = -0.1
p = 1
q = 2
x = np.linspace(0, np.pi / 2, n + 2)

# Define function f(x)
def f(x):
    return -np.cos(x)

# Create matrix A (1x1)
A = np.array([2 + q * h**2])
print("Matrix A:", A)

# Create matrix B
b = np.array([(1 + h * p / 2) * g_0 + (h**2) * f(x[1])])
print("Matrix B:", b)

from scipy.linalg import solve
U = solve(A, b)
print("Finite Difference Solution:", U)

# Construct the full solution including boundary conditions
U_full = np.concatenate(([g_0], U, [g_1]))

# Shooting Method
# Define equations and boundary conditions
def z_p(x, y, z):
    return z + 2 * y + np.cos(x)

def y_p(z):
    return z

y0 = -0.3
y1 = -0.1
n_shoot = 2
h_shoot = np.pi / 4  # Step size for shooting method
x_shoot = np.linspace(0, np.pi / 2, n_shoot + 1)

# Initial guesses for z0
z0_g1 = np.pi / 4
z0_g2 = np.pi / 2

# Solve IVP for a given initial slope z0
def solve_ivp(z0):
    yvalues = [y0]
    zvalues = [z0]

    for i in range(n_shoot):
        z_next = zvalues[i] + h_shoot * z_p(x_shoot[i], yvalues[i], zvalues[i])
        y_next = yvalues[i] + h_shoot * y_p(zvalues[i])
        zvalues.append(z_next)
        yvalues.append(y_next)

    return yvalues[-1]

# Secant Method
tol = 1e-6
max_i = 100

f1 = solve_ivp(z0_g1) - y1
f2 = solve_ivp(z0_g2) - y1

for i in range(max_i):
    z_new = z0_g2 - f2 * (z0_g2 - z0_g1) / (f2 - f1)  # Update secant
    f_new = solve_ivp(z_new) - y1  # Evaluate BVP with new guess
    if abs(f_new) < tol:  # Check for convergence
        z0_correct = z_new
        break
    z0_g1, z0_g2 = z0_g2, z_new
    f1, f2 = f2, f_new

# Solve IVP with correct z0
yvalues = [y0]
zvalues = [z0_correct]

for i in range(n_shoot):
    z_next = zvalues[i] + h_shoot * z_p(x_shoot[i], yvalues[i], zvalues[i])
    y_next = yvalues[i] + h_shoot * y_p(zvalues[i])
    zvalues.append(z_next)
    yvalues.append(y_next)

# Print results
print(f"Correct initial slope (z0): {z0_correct}")
print("x-values:", x_shoot)
print("y-values:", yvalues)

# Plot the solutions
mpl.plot(x, U_full, marker='o', linestyle='-', color='b', label='Finite Difference Solution')
mpl.plot(x_shoot, yvalues, marker='o', linestyle='-', color='g', label='Shooting Method Solution')
mpl.xlabel('x')
mpl.ylabel('U')
mpl.title(f"Solution of BVP (h=pi/4)")
mpl.legend()
mpl.grid(True)
mpl.show()
