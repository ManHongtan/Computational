import numpy as np
import matplotlib.pyplot as plt

# Constants
hbar = 1  # Planck's constant (reduced)
m = 1     # Mass of the particle
a = 1     # Width of the well (changed from 0.5 to 1)
N = 1000  # Number of points for discretization
dx = 2 * a / N  # Step size

# Discretized domain
x = np.linspace(-a, a, N)

# Define the shooting method
def solve_schrodinger(E):
    """
    Solve the Schrödinger equation for a given energy E.
    Returns the wavefunction ψ(x).
    """
    # Initialize wavefunction and its derivative
    psi = np.zeros_like(x)
    psi[1] = 1e-5  # Small non-zero initial value to start propagation
    
    # Propagate using finite difference
    for i in range(1, N - 1):
        psi[i + 1] = (2 * (1 - m * E * dx**2 / hbar**2) * psi[i] - psi[i - 1])
    
    return psi

# Boundary condition error
def boundary_condition_error(E):
    """
    Compute the boundary condition error at x = a for a given energy E.
    """
    psi = solve_schrodinger(E)
    return psi[-1]

def find_eigenvalues(num_levels):
    """
    Find the first `num_levels` eigenvalues using the shooting method.
    """
    eigenvalues = []
    guesses = np.linspace(1, 200, 10000)  # More energy guesses with a larger range
    
    for i in range(len(guesses) - 1):
        error1 = boundary_condition_error(guesses[i])
        error2 = boundary_condition_error(guesses[i + 1])
        
        # Check if there's a sign change (crossing zero)
        if error1 * error2 < 0:
            eigenvalues.append((guesses[i] + guesses[i + 1]) / 2)
        
        if len(eigenvalues) == num_levels:  # Stop if enough eigenvalues are found
            break

    # Check if we found enough eigenvalues
    if len(eigenvalues) < num_levels:
        raise ValueError(f"Failed to find {num_levels} eigenvalues. Found only {len(eigenvalues)}.")
    
    return eigenvalues


# Analytical eigenvalues
def En(n, a):
    return (hbar**2 * np.pi**2 * n**2) / (8 * m * a**2)

# Compute eigenvalues
num_levels = 6
numerical_eigenvalues = find_eigenvalues(num_levels)
analytical_eigenvalues = [En(n, a) for n in range(1, num_levels + 1)]

# Display side-by-side comparison
print(f"{'Level':<10}{'Analytical (E_n)':<20}{'Shooting Method (E_n)':<20}")
for n in range(num_levels):
    print(f"{n+1:<10}{analytical_eigenvalues[n]:<20.6f}{numerical_eigenvalues[n]:<20.6f}")

# Plot the wavefunctions for the first few eigenstates
plt.figure(figsize=(10, 6))
for i, E in enumerate(numerical_eigenvalues):
    psi = solve_schrodinger(E)
    plt.plot(x, psi / np.max(np.abs(psi)), label=f"n={i + 1}, E={E:.3f}")

plt.xlabel("x")
plt.ylabel("ψ(x) (normalized)")
plt.title("Wavefunctions for the Infinite Square Well")
plt.legend()
plt.grid()
plt.show()
