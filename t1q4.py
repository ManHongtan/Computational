import numpy as np
import matplotlib.pyplot as mpl

# Define the exact solution
def exact_solution(x):
    return np.log((np.e - 1) * x + 1)

# Define the ODE system
def u_dp(u_p):
    return -u_p**2

# Newton-Raphson to solve the IVP
def solve_ivp_newton(s, h, tol=1e-6, max_i=100):
    n = int(1 / h) + 1  
    x = np.linspace(0, 1, n)
    u = np.zeros(n)
    u_p = np.zeros(n)

    # Initial conditions
    u[0] = 0  
    u_p[0] = s

    for i in range(1, n):
        u_guess = u[i - 1] + h * u_p[i - 1]
        u_p_guess = u_p[i - 1] + h * u_dp(u_p[i - 1])

        # Newton-Raphson iterations
        for iteration in range(max_i):
            f1 = u_guess - u[i - 1] - h * u_p[i - 1]
            f2 = u_p_guess - u_p[i - 1] - h * u_dp(u_p_guess)

            # Update guesses using Newton-Raphson
            df1_du = 1  
            df1_du_prime = -h  
            df2_du_prime = 1 - h * (-2 * u_p_guess)  

            # Solve for delta corrections
            delta_u = -f1 / df1_du
            delta_u_prime = -f2 / df2_du_prime

            # Update guesses
            u_guess += delta_u
            u_p_guess += delta_u_prime

            # Check convergence
            if abs(delta_u) < tol and abs(delta_u_prime) < tol:
                break

        u[i] = u_guess
        u_p[i] = u_p_guess

    return x, u

# Secant method for root-finding
def secant_method(tol=1e-6, max_i=100):
    s0, s1 = 0.5, 1.0  
    h = 0.2  

    for iteration in range(max_i):
        # Solve the IVP for the current guesses
        _, u0 = solve_ivp_newton(s0, h)
        _, u1 = solve_ivp_newton(s1, h)

        # difference from boundary condition u(1) = 1)
        r0 = u0[-1] - 1
        r1 = u1[-1] - 1

        # Secant update
        s_new = s1 - r1 * (s1 - s0) / (r1 - r0)

        # Check for convergence
        if abs(s_new - s1) < tol:
            return solve_ivp_newton(s_new, h)

        # Update guesses
        s0, s1 = s1, s_new

# Solve the BVP using the secant method
x, u_numeric = secant_method()

# Compute the exact solution
u_exact = exact_solution(x)

# Plot the results
mpl.figure(figsize=(8, 6))
mpl.plot(x, u_numeric, 'o-', label='Numerical Solution')
mpl.plot(x, u_exact, '--', label='Exact Solution')
mpl.xlabel('x')
mpl.ylabel('u(x)')
mpl.title('Non-linear BVP(shooting method)')
mpl.legend()
mpl.grid()
mpl.show()

# Tabulate the results
print("x\tNumerical\tExact\t\tError")
for i in range(len(x)):
    print(f"{x[i]:.2f}\t{u_numeric[i]:.6f}\t{u_exact[i]:.6f}\t{abs(u_numeric[i] - u_exact[i]):.6e}")
