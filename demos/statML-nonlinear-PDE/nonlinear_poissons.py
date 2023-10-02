import numpy as np, matplotlib.pyplot as plt
import time
import sympy as sym, os

results_folder = os.path.join(os.getcwd(), "results")
if not os.path.exists(results_folder):
    os.mkdir(results_folder)

# CONFIGURATION SETTINGS
# --------------------------------------------------------
nB = 300 # num boundary points
nI = 1000 # num interior points in Omega = (0,1)^2

G_sigma = 0.2 # kernel hyperparameter
n_newton = 5 # num Newton iterations
eta = 1e-5

# ----------------------------------------------------------

# COLLOCATION POINTS AND FORCES
# ----------------------------------------------------------

# total number of opt pts (for linear operators)
N = 2 * nI + nB 
# randomly generated boundary points
xB = np.random.rand(nB,2)
for i in range(nB):
    grp = np.floor(i * 4 / nB) # 0,1,2,3 groups
    if grp == 0: # y = 0
        xB[i,1] = 0.0
    elif grp == 1: # x = 1
        xB[i,0] = 1.0
    elif grp == 2: # y = 1
        xB[i,1] = 1.0
    else: # x = 0
        xB[i,0] = 0.0

# randomly generated interior points
xI = np.random.rand(nI, 2)

# full set of points for linear operators
X = np.concatenate([xI, xB, xI], axis=0)
print(f"X shape = {X.shape}")

# get the forcing terms and forcing function
def ustar(x): # true solution
    x1 = x[0]; x2 = x[1]
    return np.sin(np.pi * x1) * np.sin(np.pi * x2) + 4 * np.sin(4 * np.pi * x1) * np.sin(4 * np.pi * x2)
x_sym = [sym.Symbol(f"x{i}") for i in range(2)]
x1 = x_sym[0]; x2 = x_sym[1]
ustar_sym = sym.sin(np.pi * x1) * sym.sin(np.pi * x2) + 4 * sym.sin(4 * np.pi * x1) * sym.sin(4 * np.pi * x2)
f_sym = sum([sym.diff(ustar_sym, x_sym[i], 2) for i in range(2)]) + ustar_sym**3

# true solution at each point
u_truth = np.array([ustar(X[i,:]) for i in range(N)])

def force(x):
    # use symbolic manipulation to get the force from the true solution
    # alternative might be to use complex-step method
    x1 = x[0]; x2 = x[1]
    return f_sym.subs(x_sym[0], x1).subs(x_sym[1], x2)

F_int = np.array([float(force(xI[i,:])) for i in range(nI)])

# plot the boundary and interior points
print("Plotting the collocation points on interior and boundary..")
plt.figure("collocationPts")
plt.plot(xB[:,0], xB[:,1], "bo", label="bndry")
plt.plot(xI[:,0], xI[:,1], "ko", label="inner")
plt.legend()
plt.savefig("results/init_colloc_pts.png")
plt.close("collocationPts")

# BUILD THE COVARIANCE / KERNEL MATRICES
# ------------------------------------------------------------------

def kernel(xm, xl, sigma=1, order=0):
    D = xm.shape[0]
    norm_sq = np.dot(xm-xl, xm-xl)
    if order == 0: # regular function
        return np.exp(-norm_sq/2.0/sigma**2)
    elif order == 1: # first Laplacian
        K = kernel(xm,xl,sigma=sigma, order=0)
        return K * (norm_sq - D * sigma**2) / sigma**4
    elif order == 2:
        K = kernel(xm,xl,sigma=sigma, order=0)
        return K * (norm_sq**2 - 2 * (2 + D) * norm_sq * sigma**2 + D * (2 + D) * sigma**4) / sigma**8

# construct cost matrix K(phi,phi) = Kpp and K(X,phi) = Kxp matrix
print("Building K(x,phi) and K(phi,phi) matrices...")
start_Theta = time.time()
Kpp = np.zeros((N,N))
Kxp = np.zeros((N,N))
for m in range(N):
    xm = X[m,:]
    m_2nd = m >= nB + nI # last group of pts, boolean for 2nd order operator
    for l in range(N):
        xl = X[l,:]
        l_2nd = l >= nB + nI
        deriv_order = 1.0 * (m_2nd + l_2nd)
        Kpp[m,l] = kernel(xm,xl,sigma=G_sigma, order=deriv_order)
        Kxp[m,l] = kernel(xm,xl,sigma=G_sigma, order=1.0 * l_2nd)

# add nugget to Kpp to improve condition number otherwise Kpp
# has det(Kpp) = 0 and the Gauss-Newton iterations do not converge (see Appendix A.1 in Y. Chen's paper)
# here I add the matrix eta * R to Kpp where R is an adaptive block-diagonal version of I
# since the omag of each block-diagonal section is different
trace11 = np.trace(Kpp[:nI, :nI])
trace22 = np.trace(Kpp[nI:nI+nB, nI:nI+nB])
trace33 = np.trace(Kpp[nI+nB:, nI+nB:])
Rvec = np.concatenate([
    np.ones((nI,)),
    (trace22/trace11) * np.ones((nB,)),
    (trace33/trace11) * np.ones((nI,))
])
R = np.diag(Rvec) # block-diagonal scaled nugget matrix
Kpp += eta * R

detKpp = np.linalg.det(Kpp)
print(f"det Kpp = {detKpp}", flush=True)

dt_Theta = time.time() - start_Theta
print(f"Built K(x,phi) and K(phi,phi) matrices in {dt_Theta} seconds.")
print(f"Inverting K(phi,phi) matrix...")
start_inv = time.time()
Kpp_inv = np.linalg.inv(Kpp)
delta_inv = time.time() - start_inv
print(f"K(phi,phi) = Theta matrix ({N},{N}) inverted in {delta_inv} seconds.")

# ------------------------------------------------------------------------

# NEWTON ITERATION
# ------------------------------------------------------------------------
# Newton iterations for optimal w weights
# could potentially store the sequence of w values if I want
print("Performing the Newton's iterations...")
start_newton = time.time()
w = np.zeros((nI,))

# useful functions
def z_func(w):
    z = np.concatenate([w, np.zeros((nB,)), F_int - w**3], axis=0)
    return np.reshape(z, newshape=(N,1))

def cost_func(w):
    z = z_func(w)
    return z.T @ Kpp_inv @ z

for inewton in range(n_newton):
    z = z_func(w)
    icost = cost_func(w)
    
    # Jacobian matrix J_ij = dzj/dwi = nabla_w z has dimension R^{s(w) x s(z)}
    dz_dw = np.concatenate([np.eye(nI), np.zeros((nI, nB)), -np.diag(3*w**2)], axis=-1)

    # Newton's iteration update dw = - Hinv * grad of objective
    # with matrix H = dz_dw * Kpp^-1 * dz_dw^T in R^(s(w) x s(w))
    H = 2 * dz_dw @ Kpp_inv @ dz_dw.T
    # gradient of objective is grad = dz_dw * Kpp_inv * z in R^(s(w) x 1)
    grad = 2 * dz_dw @ Kpp_inv @ z

    # sanity check for invertibility of Kpp
    # x = Kpp_inv @ z
    # x = np.linalg.solve(Kpp, z)
    # resid = Kpp @ x - z
    # zresid = np.max(np.abs(resid))
    # print(f"det of kpp = {np.linalg.det(Kpp)}")
    # print(f"zresid vec = {resid[:4]} z = {z[:4]} shape = {resid.shape}")
    # print(f"zresid = {zresid} on linear solve")

    # DERIVATIVE TEST FOR COST GRADIENT
    # # compare gradient to very small change in w
    # wpert = np.random.rand(nI)
    # wpert = np.zeros((nI,))
    # wpert[0] = 1.0
    # adj_deriv = np.dot(grad[:,0], wpert)
    # h = 1e-30
    # cost = cost_func(w + wpert * h * 1j)
    # cs_deriv = float(np.imag(cost) / h)
    # rel_error = (adj_deriv - cs_deriv) / cs_deriv
    # print(f"cost grad adj deriv = {adj_deriv}")
    # print(f"cost grad cs deriv = {cs_deriv}")
    # print(f"cost grad rel error = {rel_error}", flush=True)
    # exit()

    # perform the Newton's update on w
    dw = - np.linalg.solve(H, grad)

    # compute the new objective cost
    newz = z + dz_dw.T @ dw
    cost = newz.T @ Kpp_inv @ newz
    print(f"cost_{inewton} = {float(cost):.4e}")

    # check if this cost is the local linearized minimum
    # for i in range(10):
    #     wpert = np.random.rand(nI,1) * 1e-4
    #     zpert = dz_dw.T @ wpert
    #     testz = z + zpert
    #     test_cost = testz.T @ Kpp_inv @ testz
    #     test_cost_ratio = float(test_cost) / float(cost)
    #     print(f"\ttest cost_{i} = {float(test_cost):.4e}, ratio = {test_cost_ratio:.4e}")

    # unity learning rate
    w += dw[:,0]

z = np.concatenate([w, np.zeros((nB,)), F_int - w**3], axis=0)
z = np.reshape(z, newshape=(N,1))

# compute the solution vector u = Kxp * Kpp^-1 * z
u = Kxp @ Kpp_inv @ z
u = np.reshape(u, newshape=(N,))

dt_newtons = time.time() - start_newton
print(f"finished newton's iteration in {dt_newtons} sec.")

# DONE with NEWTON'S METHOD
# ------------------------------------------------------------------------

# PLOT AND REPORT SOLUTION
# -------------------------------------------------------------------------
plt.figure("nonlinear-poissons")
plt.scatter(X[:,0], X[:,1], c=u)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.savefig("results/nonlinear-poisson.png", dpi=300)
plt.close("nonlinear-poissons")

# compute the solution residuals and plot them
uresid = u - u_truth
resid_norm = np.max(np.abs(uresid))
rel_resid_norm = resid_norm / np.max(np.abs(u_truth))
print(f"u residual norm = {rel_resid_norm:.4e}")

# plot the u solution error
plt.figure("nonlinear-poissons-error")
plt.scatter(X[:,0], X[:,1], c=uresid)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.savefig("results/nonlinear-poisson-error.png", dpi=300)
plt.close("nonlinear-poissons-error")

# INTERPOLATE 2D MESH 
# ----------------------------------------------------------------------
# make a 2D mesh with meshgrid
ngrid = 30
ngrid_sq = ngrid**2
sigma_n = 1e-4
x1vec = np.linspace(0,1,30)
x2vec = x1vec.copy()
X1, X2 = np.meshgrid(x1vec, x2vec)
# goes by row 1, row 2, etc. after flattened
X1flat = np.reshape(X1, newshape=(ngrid_sq,1))
X2flat = np.reshape(X2, newshape=(ngrid_sq,1))
Xstar = np.concatenate([X1flat, X2flat], axis=1)
#print(f"X1 = {X1[:5,:5]} shape = {X1.shape}, X1flat = {X1flat[:5]}, shape = {X1flat.shape}")
print("Building kernels K(X*,X) and K(X,X) for interpolating 2d mesh..")
Kxstar_x = np.array([[kernel(Xstar[i,:], X[j,:], sigma=G_sigma) for j in range(N)] for i in range(ngrid_sq)])
Kxx = np.array([[kernel(X[i,:], X[j,:], sigma=G_sigma) for j in range(N)] for i in range(N)])
print("Done building kernels")
print("Solving kernel inverse problem..")
ucol = np.reshape(u, newshape=(N,1))
ustar = Kxstar_x @ np.linalg.solve(Kxx+sigma_n**2 * np.eye(N), ucol)
print("Done solving kernel inverse proble, now plotting..")
X1plot = np.reshape(Xstar[:,0], (ngrid,ngrid))
X2plot = np.reshape(Xstar[:,1], (ngrid,ngrid))
uplot = np.reshape(ustar[:,0], (ngrid,ngrid))

# plot the interpolated solution
plt.figure("u-2d-mesh")
fig, ax = plt.subplots(1, 1)
ax.contourf(X1plot, X2plot, uplot, levels=50)
plt.plot(X[:,0], X[:,1], "kx", markersize=3)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.savefig("results/nonlinear-poissons-2dmesh.png")
plt.close("u-2d-mesh")