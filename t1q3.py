import numpy as np
import matplotlib.pyplot as mpl

# Define parameters
h = 0.2  
a, b = 1.0, 2.0 
alpha = 1/2  
beta = 1/3  

n = int((b - a) / h) + 1  
x = np.linspace(a, b, n) 

# Initialize u using the linear gradient as the initial guess
u = np.zeros(n)
m = (beta - alpha) / (b - a)
for i in range(n):
    u[i] = alpha + i * h * m

# Nonlinear function f(u) and its derivative
def f(u):
    return u**3

def f_prime(u):
    return 3 * u**2

# Construct the tridiagonal matrix for the linearized system
def construct_matrix(u, h):
    diag = np.zeros(n - 2)  
    lower = np.ones(n - 3) 
    upper = np.ones(n - 3)  

    for i in range(n - 2):
        diag[i] = -2 - h**2 * f_prime(u[i + 1])

    return diag, lower, upper

# Solve tridiagonal system using diagonalization
def solve_tridiagonal(diag, lower, upper, rhs):
    A = np.diag(diag) + np.diag(lower, -1) + np.diag(upper, 1)
    eigvals, eigvecs = np.linalg.eigh(A)  
    rhs_transformed = eigvecs.T @ rhs  
    x_transformed = rhs_transformed / eigvals  
    x = eigvecs @ x_transformed  
    return x

# Newton-Raphson iteration
tol = 1e-6
max_i = 100

for i in range(max_i):
    u_old = u.copy()
    rhs = np.zeros(n - 2)
    for i in range(1, n - 1):
        rhs[i - 1] = u[i - 1] - 2 * u[i] + u[i + 1] - h**2 * f(u[i])

    diag, lower, upper = construct_matrix(u, h)  # Construct the tridiagonal matrix
    delta_u = solve_tridiagonal(diag, lower, upper, -rhs)   # Solve the tridiagonal system using diagonalization
    u[1:-1] += delta_u

    # Check for convergence
    if np.linalg.norm(delta_u, ord=np.inf) < tol:
        break

# Exact solution for comparison
def exact_solution(x):
    return 1.0 / (x + 1)

u_exact = exact_solution(x)

# Plot
mpl.figure(figsize=(8, 6))
mpl.plot(x, u, 'o-', label='Numerical Solution')
mpl.plot(x, u_exact, '--', label='Exact Solution')
mpl.xlabel('x')
mpl.ylabel('u(x)')
mpl.title('Non-linear BVP(finite difference)')
mpl.legend()
mpl.grid()
mpl.show()

# Print results for comparison
print("x\tNumerical\tExact\t\tError")
for i in range(n):
    print(f"{x[i]:.2f}\t{u[i]:.6f}\t{u_exact[i]:.6f}\t{abs(u[i] - u_exact[i]):.6e}")
