import sympy as sym
import numpy as np
# compute derivatives of Gaussian Kernel from Yifan Chen paper on Gaussian Processes for nonlinear PDEs, 2021
D = 1
x = np.array([sym.Symbol(f"x{i}") for i in range(D)])
xp = np.array([sym.Symbol(f"xp{i}") for i in range(D)])
sigma = sym.Symbol("sigma")
xdiff = x - xp
xsq = np.dot(xdiff, xdiff)
print(f"xsq = {xsq}")
kernel = sym.exp(-xsq / 2 / sigma**2)
print(f"kernel = {kernel}")

# gradient
Kgrad = [sym.diff(kernel, f"x{i}") for i in range(3)]
K_lap = sum([sym.diff(Kgrad[i], f"x{i}") for i in range(3)])
#print(f"Kgrad = {Kgrad}")
#print(f"Klapl = {K_lap}")
K_grad2 = [sym.simplify(Kgrad[i] / kernel) for i in range(3)]
K_lap2 = sym.simplify(K_lap / kernel)
print(f"Kgrad = K * {K_grad2}")
print(f"Klap = K * {K_lap2}")

K_double_grad = [sym.diff(K_lap, f"xp{i}") for i in range(3)]
K_double_grad2 = [sym.simplify(K_double_grad[i] / kernel) for i in range(3)]
K_double_lapl = sum([sym.diff(K_double_grad[i], f"xp{i}") for i in range(3)])
K_double_lapl2 = sym.simplify(K_double_lapl / kernel)
print(f"K_lapl_grad = K * {K_double_grad2}")
print(f"K_double_lapl = K * {K_double_lapl2}")

# alternative expression, compare them
K_double_lapl2_compare = np.dot(xdiff,xdiff)**2 / sigma**8 - 2 * np.dot(xdiff,xdiff) * (2+D) / sigma**6 + D*(2+D) / sigma**4
K_double_lapl_residual = sym.simplify(K_double_lapl2 - K_double_lapl2_compare)
print(f"K double lapl residual = {K_double_lapl_residual}")