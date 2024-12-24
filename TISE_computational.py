#!/usr/bin/env python
# coding: utf-8

# Include all necessary libraries and define given constant and potential well. The constants are reduced to normalised unit to simplify calculations (hbar=1, m=1). The parameters alpha and lamda(lam) determine the spatial characteristic and depth of the potential well respectively.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# Constants
hbar = 1.0
m = 1.0
alpha = 1.0
lam = 4.0


# The potential is a hyperbolic cosine with symmetric well shape. The particle is confined within the potential well in bound state of E<V(max)

# In[ ]:


# Potential definition
def V(x):
    return (hbar**2 * alpha**2 * lam * (lam - 1) * (0.5 - 1 / (np.cosh(alpha * x)**2))) / (2 * m)


# f(E) computes the mismatch of wavefunction deriatives at turning points for given energy (E). Continuity conditions apply for the wavefunction where:
# *psi(left)=psi(right), psi'(left)=psi'(right)*

# In[ ]:


# Function f(E)
def f_E(E):
    def k_squared(E, V):
        return -2 * m / hbar ** 2 * (E - V)


# k_squared(E,V) is the squared wave number obtain from the Schrodinger equation. Numerov method is applied for solving the second-order differential equation by using three consecutive points to compute the next value. Using Numerov method, the wavefunction is integrated from the left and right boundary.

# In[ ]:


def Numerov_left(y_left, h, E, V_x):
    k2 = k_squared(E, V_x)
    y = np.zeros(n + 1)
    y[0] = y_left
    y[1] = y_left + 1e-10
    for i in range(1, n):
        y[i + 1] = (y[i] * (2 + 10 / 12 * h ** 2 * k2[i]) - y[i - 1] * (1 - h ** 2 / 12 * k2[i - 1])) / (1 - h ** 2 / 12 * k2[i + 1])
    return y

def Numerov_right(y_right, h, E, V_x):
    k2 = k_squared(E, V_x)
    y = np.zeros(n + 1)
    y[-1] = y_right
    y[-2] = y_right + 1e-10
    for i in range(1, n):
        y[-(i + 2)] = (y[-(i + 1)] * (2 + 10 / 12 * h ** 2 * k2[-(i + 1)]) - y[-i] * (1 - h ** 2 / 12 * k2[-i])) / (1 - h ** 2 / 12 * k2[-(i + 2)])
    return y


# Turning point is when E=V where the wavefunction transform between oscillatory and exponential behaviour. It is determined using the Bisection Method.

# In[ ]:


# Find turning points
a = 0
b = 15 / alpha
tol = 1e-15
max_iter = 1000

def f(x, E):  # Function for turning points
    return E - V(x)

def bisection(f, a, b, E, tol, max_iter):
    if f(a, E) * f(b, E) > 0:
        raise ValueError('Function has same signs at both endpoints of the interval.')
    for _ in range(max_iter):
        c = (a + b) / 2
        fc = f(c, E)
        if abs(fc) < tol:
            return c
        elif f(a, E) * fc < 0:
            b = c
        else:
            a = c
    return (a + b) / 2

left_root = -bisection(f, a, b, E, tol, max_iter)
right_root = bisection(f, a, b, E, tol, max_iter)


# The shooting method sets up the spatial grid and computes the potential at each point then integrates the wavefunctions from both boundaries and calculates the mismatch at the turning point.

# In[ ]:


# Shooting method setup
midpoint = (right_root + left_root) / 2
width = (right_root - left_root) / 2
x_far_left = midpoint - 10 * width
x_far_right = midpoint + 10 * width

n = 20000
h = (x_far_right - x_far_left) / n
x = np.linspace(x_far_left, x_far_right, n + 1)

V_x = V(x)

Left_points = Numerov_left(0, h, E, V_x)
Right_points = Numerov_right(0, h, E, V_x)

n_turning = int((right_root - x_far_left) / h)
return ((Left_points[n_turning + 1] - Left_points[n_turning - 1]) / (2 * h * Left_points[n_turning]) - 
        (Right_points[n_turning + 1] - Right_points[n_turning - 1]) / (2 * h * Right_points[n_turning]))


# Here the code defines an energy range, E_min to E_max and then divides into intervals. The range is determined by examining the potential range. Sign changes in f(E) is checked across intervals to indicate potential eigenvalues. For every interval with sign change, it uses the Brent's method (root_scalar) to find the root (eigenvalues)

# In[ ]:


# Finding eigenvalues
E_min, E_max = -2.99, 2.99
num_intervals = int((E_max - E_min) / 0.01)
E_values = np.linspace(E_min, E_max, num_intervals + 2)

roots = []
for i in range(len(E_values) - 1):
    a, b = E_values[i], E_values[i + 1]
    if f_E(a) * f_E(b) < 0:  # Sign change indicates a root
        try:
            sol = root_scalar(f_E, bracket=[a, b], method='brentq')
            if sol.converged:
                roots.append(sol.root)
        except ValueError:
            pass

# Remove duplicates and display results
unique_roots = np.unique(np.round(roots, decimals=15))


# Visualisation of the potential V(x) and quantized eigenvalues E(n). The plot of f(E) changes with energy where roots of f(E)=0 confirms the eigenvalues. Eigenvalues lie between potential maximum.

# In[ ]:


print("First eigenvalues found:")
for i, root in enumerate(unique_roots[:6]):  # Show only the first 6
    print(f"Eigenvalue {i + 1}: {root:.6f}")

# Plot the potential and eigenvalues
x = np.linspace(-10, 10, 1000)
plt.plot(x, V(x), label="Potential V(x)")
for i, root in enumerate(unique_roots[:6]):
    plt.axhline(root, linestyle='--', label=f"Eigenvalue {i + 1}: {root:.2f}")
plt.title("Potential and First 6 Eigenvalues")
plt.xlabel("x")
plt.ylabel("Energy")
plt.legend()
plt.grid()
plt.show()

# Plot f(E) to visualize roots
plt.plot(E_values, [f_E(E) for E in E_values])
plt.axhline(0, color='red', linestyle='--', label="f(E) = 0")
plt.title("Mismatch Function f(E) vs Energy")
plt.xlabel("Energy (E)")
plt.ylabel("f(E)")
plt.legend()
plt.grid()
plt.show()


# In[ ]:




