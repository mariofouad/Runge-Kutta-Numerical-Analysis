import numpy as np
import matplotlib.pyplot as plt
y_initial = -2.9499
z_initial = 9.778768

#================================================================================================
#================================== RUNGE KUNTA FOURTH DEGREE ===================================
#================================================================================================

def z_equation(x, y, z):
    return (1 / (np.cos(np.log(x)) - np.sin(np.log(x))) - x * z - 4 * y) / x**2

def k_iteration(x, y, z, h):
    k1 = z
    m1 = z_equation(x, y, z)
    k2 = z + m1 * h / 2
    m2 = z_equation(x + h / 2, y + k1 * h / 2, z + m1 * h / 2)
    k3 = z + m2 * h / 2
    m3 = z_equation(x + h / 2, y + k2 * h / 2, z + m2 * h / 2)
    k4 = z + m3 * h
    y_new = y + (k1 + 2 * (k2 + k3) + k4) * h / 6
    z_new = z + (m1 + 2 * (m2 + m3) + z_equation(x + h, y_new, k4)) * h / 6 # update z
    return y_new, z_new

def RK(x, y, z, h, x_final):
    x_values = [x]
    y_values = [y]
    while x <= x_final:
        y, z = k_iteration(x, y, z, h)
        x = x + h
        x_values.append(x)
        y_values.append(y)
    return x_values, y_values


# Get User Inputs
print("===========================================================================")
print("============== WELCOME TO RAUNG KUNTTA DIFFERENTIAL EQUATIONS =============")
print("===========================================================================")
print()

x_initial = float(input("Enter initial value of X: "))
x_final = float(input("Enter final value of X: "))
h = float(input("Enter step size: "))

# Call RK to get the arrays of x and y values
x_values_ARR, y_values_ARR = RK(x_initial, y_initial , z_initial, h, x_final)
y_at_5 = y_values_ARR[-1]

#Plot RK graph
plt.plot(x_values_ARR, y_values_ARR)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solution of Differential Equation using RK fourth degree')
plt.grid(True)
plt.show()

#================================================================================================
#======================================= RUNGE KUNTA NYSTRON ====================================
#================================================================================================
def RKN_iteration(x, y, z, h):
    k1 = 0.5 * h * z_equation(x, y, z)
    k = 0.5 * h * (z + 0.5 * k1)
    k2 = 0.5 * h * z_equation(x + 0.5 * h, y + k, z + k1)
    k3 = 0.5 * h * z_equation(x + 0.5 * h, y + k, z + k2)
    l = h * (z + k3)
    k4 = 0.5 * h * z_equation(x + h, y + l, z + 2 * k3)
    y_new = y + h * (z + (k1 + k2 + k3) * 1/3)
    z_new = z + (k1 + 2 * k2 + 2 * k3 + k4) * 1/3
    return y_new, z_new


def RKN(x, y, z, h, x_final):
    x_values = [x]
    y_values = [y]
    while x <= x_final:
        y, z = RKN_iteration(x, y, z, h)
        x = x + h
        x_values.append(x)
        y_values.append(y)
    return x_values, y_values

x_values_arr, y_values_arr  = RKN(x_initial, y_initial, z_initial, h, x_final)

#Plot RKN graph
plt.plot(x_values_arr, y_values_arr)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solution of Differential Equation using RKN')
plt.grid(True)
plt.show()

#================================================================================================
#============================================= EXACT ============================================
#================================================================================================
def y_exact(x):
    def sec(x):
        return 1 / np.cos(x)

    term1 = np.sin(2 * np.log(x)) * (
        (np.log(sec(4 * np.log(x)) + np.tan(4 * np.log(x)))) / 16 +
        np.tan(4 * np.log(x)) +
        (np.log(x) / 4) +
        (np.log(sec(4 * np.log(x))) / 16) -
        (3 / 2)
    )

    term2 = np.cos(2 * np.log(x)) * (
        (np.log(x) / 4) -
        (np.log(sec(4 * np.log(x)) + np.tan(4 * np.log(x)))) / 16 -
        (np.log(sec(4 * np.log(x))) / 16)
    )

    return term1 + term2
print(y_exact(5))
x_values_exact = np.arange(x_initial, x_final + h, h)
y_values_exact = y_exact(x_values_exact)

# Plot three graphs calculations
plt.figure(figsize=(10, 6))
plt.plot(x_values_ARR, y_values_ARR, label='RK')
plt.plot(x_values_arr, y_values_arr, label='RKN')
plt.plot(x_values_exact, y_values_exact, label='Exact')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Exact VS RK VS RKN')
plt.legend()
plt.grid(True)
plt.show()