import numpy as np
import matplotlib.pyplot as plt

h = 1/1000

# Define the differential equation
def f(x, y):
    return y * (1 - np.exp(2 * x))

# Initial conditions
x0 = 0
y0 = 1

# Euler Method
x_values = [x0]
y_values = [y0]
x = x0
y = y0

for i in range(int(1/h) + 1):
    print("x = ", x, "y = ", y)
    y = y + h * f(x, y)
    x = x + h
    x_values.append(x)
    y_values.append(y)

plt.scatter(x_values, y_values, color='blue', label='Euler Method')

# Runge-Kutta 2nd Order (RK2) Method
x0 = 0
y0 = 1

x_values2 = [x0]
y_values2 = [y0]
x = x0
y = y0

# Define K2 function for RK2
def K2(x, y):
    return f(x + h, y + h * f(x, y))

for i in range(int(1/h) + 1):
    print(f'x = {x}, y = {y}')
    y = y + h / 2 * (f(x, y) + K2(x, y))
    x = x + h
    x_values2.append(x)
    y_values2.append(y)

plt.scatter(x_values2, y_values2, color='red', label='RK2 Method')

# Runge-Kutta 4th Order (RK4) Method
def K2(x, y):
    return f(x + h / 2, y + h / 2 * f(x, y))

def K3(x, y):
    return f(x + h / 2, y + h / 2 * K2(x, y))

def K4(x, y):
    return f(x + h, y + h * K3(x, y))

x0 = 0
y0 = 1

x_values3 = [x0]
y_values3 = [y0]

x = x0
y = y0
for i in range(int(1/h) + 1):
    print(f'x = {x}, y = {y}')
    y = y + (f(x, y) + 2 * K2(x, y) + 2 * K3(x, y) + K4(x, y)) * h / 6
    x = x + h
    x_values3.append(x)
    y_values3.append(y)

plt.scatter(x_values3, y_values3, color='yellow', label='RK4 Method', alpha=0.5)

plt.grid()
plt.title(f'Euler Method vs RK2 vs RK4 Methods with $h={h}$')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

#Discuss the accuracy for each computation:
#Euler Method: The simplest method, offering quick and easy approximations, but with lower accuracy. Its error grows quickly with larger step sizes, making it less reliable over long intervals.
#RK2: Provides a better balance between accuracy and computation compared to Euler. It's more precise but still not as accurate as higher-order methods, suitable for moderate accuracy needs.

#RK4: The most accurate of the three methods, ideal for high-precision solutions. It handles larger step sizes well but requires more computational effort due to its complexity.
