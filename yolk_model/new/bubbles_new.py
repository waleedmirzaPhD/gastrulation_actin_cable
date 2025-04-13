import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import jacfwd
from jax import config
config.update("jax_enable_x64", True)

# Constants
gamma_e = 1.0   # tension of EVL
eta_e = 1.0     # viscosity of EVL
gamma_y = 1.0
L = 1./7.       # width of the actin cable
V = 4*jnp.pi/3. # Fixed volume
#-------------------

#CABLE
gamma = 5     # active tension in the cable
eta = 1.E-1        # viscosity of actin cable
l = 1*L          # hydrodynamic length in the cable
#-------------------

#Time integrator
dt = 1.E-6     # Time step
max_iter = 5   # Max Newton iterations
tol = 1e-6     # Convergence tolerance
#-------------------

def base_radius(h, H):
    a = jnp.sqrt((6 * V / jnp.pi - h**3 - H**3) / (3 * (h + H)))
    return a

def compute_force(f_dot_h_H, dt):
    Force =  gamma * dt * (1-1/jnp.cosh(L/l)) - f_dot_h_H * eta / l *jnp.tanh(L/l)
    return Force

def equations(h, H, h_N, H_N, dt):

    a_N = base_radius(h_N, H_N)
    a = base_radius(h, H)

    dot_h = h-h_N
    dot_H = H-H_N
    dot_a = a-a_N

    R_e = (a**2 + h**2) / (2 * h)
    R_y = (a**2 + H**2) / (2 * H)
    R_y_N = (a_N**2 + H_N**2) / (2 * H_N)

    theta_e = jnp.arccos((h-R_e)/R_e)
    theta_y = jnp.arccos((H-R_y)/R_y)
    theta_y_N = jnp.arccos((H_N-R_y_N)/R_y_N)

    A_e = jnp.pi * (a**2 + h**2)
    dot_A_e = 2 * jnp.pi * (a * dot_a + h * dot_h)

    sigma_e = gamma_e * dt + eta_e * dot_A_e / A_e   
    sigma_c = gamma * dt + eta * dot_a/a
    sigma_y = gamma_y *dt 

    # f_dot_h_H = ((a**2 + h**2) * dot_h - (a**2+H**2) * dot_H)/(2*a**2+h**2+H**2) * jnp.sin(theta_y)  # FIXME: what should we have here?!?
    f_dot_h_H = R_y * (theta_y-theta_y_N)
    Force = compute_force(f_dot_h_H, dt)

    eq1 = sigma_e * jnp.sin(theta_e) - (sigma_y + Force) * jnp.sin(theta_y)
    eq2 = sigma_e* jnp.cos(theta_e) + (sigma_y + Force) * jnp.cos(theta_y) - sigma_c*L/a
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
        # print(jnp.linalg.norm(F),jnp.linalg.norm(delta))
        if jnp.linalg.norm(F) < tol:
            solved = True
            break
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
            if(n_iter<4 and dt<1):
                dt *= 1.1
        else:
            h,H = h0,H0
            dt /= 1.1
            if(dt < 1.E-6):
                break
        # if H<0.1:
        #     # non dimensionolise t
        #     t_tot = t/eta_e * gamma_e
        #     break
    
h0, H0 = 0.5, 1.5
t_max = 1000
time_integrator(h0, H0, t_max, dt, "data.txt")