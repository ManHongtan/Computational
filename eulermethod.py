import numpy as np
import matplotlib.pyplot as mpl

def f(x, y):
    return y * (1 - np.exp(2 * x))

#euler method
def euler_method(x0, y0, h, n):
    x = np.linspace(x0, 1, n+1)
    y = np.zeros(n+1)
    y[0] = y0
    for i in range(n):
        y[i+1] = y[i] + h * f(x[i], y[i])
    return x, y

#initial values
x0 = 0
y0 = 1
h_values = [1/4, 1/10, 1/100, 1/1000]
results = []

#results for each step size
for h in h_values:
    n = int(1 / h) 
    x, y = euler_method(x0, y0, h, n)
    results.append((x, y))

#plot the results
for i, h in enumerate(h_values):
    mpl.plot(results[i][0], results[i][1], label=f"h={h}")

mpl.title("Comparison of y' = y(1 - exp(2x)) approximation with different h values (Euler's Method)")
mpl.xlabel("x")
mpl.ylabel("y'")
mpl.legend()
mpl.grid(True)
mpl.show()

#print the x and y values
for i, h in enumerate(h_values):
    print(f"\nResults for h = {h}:")
    for xi, yi in zip(results[i][0], results[i][1]):
        print(f"x: {xi:.2f}, y: {yi:.6f}")
