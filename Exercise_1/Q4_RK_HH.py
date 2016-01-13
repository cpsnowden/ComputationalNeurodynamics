"""
Computational Neurodynamics
Exercise 1 Q4

(C) Murray Shanahan et al, 2015
"""

import numpy as np
import matplotlib.pyplot as plt

def xdot(x, I, fP):

    v, m, n, h = x
    gNa, gK, gL, ENa, EK, EL, C = fP

    alpha_m = (2.5 - 0.1 * v) / (np.exp(2.5 - 0.1 * v) - 1.0)
    alpha_n = (0.1 - 0.01 * v) / (np.exp(1.0 - 0.1 * v) - 1.0)
    alpha_h = 0.07 * np.exp(-v / 20.0)
    beta_m = 4.0 * np.exp(-v / 18.0)
    beta_n = 0.125 * np.exp(-v / 80.0)
    beta_h = 1.0 / (np.exp(3.0 - 0.1 * v) + 1.0)

    mdot = (alpha_m * (1 - m) - beta_m * m)
    ndot=  (alpha_n * (1 - n) - beta_n * n)
    hdot = (alpha_h * (1 - h) - beta_h * h)

    sigmaIk = gNa * (m ** 3) * h * (v - ENa) + gK * (n ** 4) * (v - EK) + gL * (v - EL)
    vdot = (-sigmaIk + I) / C

    return np.array([vdot, mdot, ndot, hdot])

def rk4(x_prev, I, dt, fP):

    k1 = xdot(x_prev, I, fP)
    k2 = xdot(x_prev + 0.5 * dt * k1, I, fP)
    k3 = xdot(x_prev + 0.5 * dt * k2, I, fP)
    k4 = xdot(x_prev + dt * k3, I, fP)

    return x_prev + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
