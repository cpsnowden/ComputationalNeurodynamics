"""
Computational Neurodynamics
Exercise 1 Q3

(C) Murray Shanahan et al, 2015
"""

import numpy as np
import matplotlib.pyplot as plt

def eul(x_prev, I ,dt, fP):

    gNa, gK, gL, ENa, EK, EL, C = fP
    v_prev, m_prev, n_prev, h_prev = x_prev
    
    alpha_m = (2.5 - 0.1 * v_prev) / (np.exp(2.5 - 0.1 * v_prev) - 1.0)
    alpha_n = (0.1 - 0.01 * v_prev) / (np.exp(1.0 - 0.1 * v_prev) - 1.0)
    alpha_h = 0.07 * np.exp(-v_prev / 20.0)
    beta_m = 4.0 * np.exp(-v_prev / 18.0)
    beta_n = 0.125 * np.exp(-v_prev / 80.0)
    beta_h = 1.0 / (np.exp(3.0 - 0.1 * v_prev) + 1.0)

    m_next = m_prev + dt * (alpha_m * (1 - m_prev) - beta_m * m_prev)
    n_next = n_prev + dt * (alpha_n * (1 - n_prev) - beta_n * n_prev)
    h_next = h_prev + dt * (alpha_h * (1 - h_prev) - beta_h * h_prev)

    sigmaIk = gNa * (m_prev ** 3) * h_prev * (v_prev - ENa) + gK * (n_prev ** 4) * (v_prev - EK) + gL * (v_prev - EL)

    v_next = v_prev + dt * (-sigmaIk + I) / C

    return np.array([v_next, m_next, n_next, h_next])