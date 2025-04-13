import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import jacfwd
from jax import config
config.update("jax_enable_x64", True)

# Constants
gamma_e = 1.0   # active tension of EVL
eta_e = 1.0     # viscosity of EVL
L = 1./7.         # width of the actin cable
V = 4*jnp.pi/3. # Fixed volume
#-------------------

#CABLE
gamma = 2    # active tension in the cable
eta = 1.0      # viscosity of actin cable
l = 0.3*L       # hydrodynamic length in the cable
#-------------------

#Time integrator
dt0 = 1.E-4     # Time step
max_iter = 5   # Max Newton iterations
tol = 1e-6     # Convergence tolerance
#-------------------

def base_radius(h, H):
    a = jnp.sqrt((6 * V / jnp.pi - h**3 - H**3) / (3 * (h + H)))
    return a

def compute_sigma_y(sigma_e, f_dot_h_H, dt):
    sigma_y = gamma*dt + (eta / (2 * l)) * f_dot_h_H * (jnp.exp(L / l) - jnp.exp(-L / l)) + ((sigma_e - gamma*dt) / 2) * (jnp.exp(L / l) + jnp.exp(-L / l))
    return sigma_y

def equations(h, H, h_N, H_N, dt):

    a_N = base_radius(h_N, H_N)
    a = base_radius(h, H)

    dot_h = h-h_N
    dot_H = H-H_N
    dot_a = a-a_N

    R_e = (a**2 + h**2) / (2 * h)
    R_y = (a**2 + H**2) / (2 * H)

    cos_theta_e = 1 - (2 * h**2) / (a**2 + h**2)
    cos_theta_y = 1 - (2 * H**2) / (a**2 + H**2)

    A_e = jnp.pi * (a**2 + h**2)
    A_e_N = jnp.pi * (a_N**2 + h_N**2)
    dot_A_e =A_e-A_e_N

    f_dot_h_H = dot_h-dot_H  # FIXME: what should we have here?!?

    sigma_e = gamma_e * dt + eta_e * dot_A_e / A_e
    sigma_c = gamma * dt + eta * dot_a/a
    sigma_y = compute_sigma_y(sigma_e, f_dot_h_H, dt) #FIXME: angles should go here

    eq1 = sigma_e / R_e - sigma_y / R_y
    eq2 = a*(sigma_e * cos_theta_e + sigma_y * cos_theta_y) + sigma_c * L
    return jnp.array([eq1, eq2])

def jacobian(h, H, h0, H0, dt):
    def equations_wrapped(vars):
        return equations(vars[0], vars[1], h0, H0, dt)
    return jacfwd(equations_wrapped)(jnp.array([h, H]))

def newton_raphson(h0, H0, dt):
    h, H = h0, H0
    solved = False
    # print("NR")
    for _ in range(max_iter):
        F = equations(h, H, h0, H0, dt)
        J = jacobian(h, H, h0, H0, dt)
        delta = jnp.linalg.solve(J, F)
        h -= delta[0]
        H -= delta[1]
        # print("    ",jnp.linalg.norm(F),jnp.linalg.norm(delta))
        if jnp.linalg.norm(F) < tol:
            solved = True
            break
    # print("---")

    return h, H, solved, _

def time_integrator(h0, H0, t_max, dt0, filename):
    t = 0.0
    dt = dt0
    h, H = h0, H0
    
    results = open(filename,"w")
    results.close()

    while t<t_max:
        h0,H0 = h,H
        h, H, solved, n_iter = newton_raphson(h, H, dt)
        if(solved):
            print(t,dt)

            results = open(filename,"a")
            results.write("{0:.8e} {1:.8e} {2:.8e} {3:.8e}\n".format(t,h,H,base_radius(h,H)))
            results.close()

            t += dt

            if(h/(h+H)>0.99):
                break
            if(n_iter<4):
                dt *= 1.1
        else:
            h,H = h0,H0
            dt /= 1.1
            if(dt < 1.E-8):
                break
    
h0, H0 = 0.5, 1.5
t_max = 10
time_integrator(h0, H0, t_max, dt0, "data.txt")
