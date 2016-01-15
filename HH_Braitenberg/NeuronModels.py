"""
Neuron Models

"""

import numpy as np

def HodgekinHuxley(x, param):

    """
    Computes the differential of the HH states given the current states and the model parameters

        :param x: Current state [v(t); m(t); n(t); h(t)]
        :type x: np.array
        :param param: Hodgkin-Huxley model parameters [I(t), gNa, gK, gL, ENa, EK, EL, C]
        :type param: np.array
        :return: dx/dt (differential of state) [(v_dt(t),m_dt(t),n_dt(t),h_dt(t))]
        :rtype: np.array

    param[]:
        | I(t): Total membane current per unit area
        | gNa: Sodium conductance per unit area
        | gK: Potassium conductance per unit area
        | gL: Leakage conductance per unit area
        | ENa: Sodium reversal potential
        | EK: Potassium reversal potential
        | EL: Leakage reversal potential
        | C: Membrane capacitance per unit area
    """
    v, m, n, h = x
    I, gNa, gK, gL, ENa, EK, EL, C = param

    alpha_m = (2.5 - 0.1 * v) / (np.exp(2.5 - 0.1 * v) - 1.0)
    alpha_n = (0.1 - 0.01 * v) / (np.exp(1.0 - 0.1 * v) - 1.0)
    alpha_h = 0.07 * np.exp(-v / 20.0)
    beta_m = 4.0 * np.exp(-v / 18.0)
    beta_n = 0.125 * np.exp(-v / 80.0)
    beta_h = 1.0 / (np.exp(3.0 - 0.1 * v) + 1.0)

    mdot = (alpha_m * (1 - m) - beta_m * m)
    ndot = (alpha_n * (1 - n) - beta_n * n)
    hdot = (alpha_h * (1 - h) - beta_h * h)

    sigmaIk = gNa * (m ** 3) * h * (v - ENa) + gK * (n ** 4) * (v - EK) + gL * (v - EL)
    vdot = (-sigmaIk + I) / C

    return np.array([vdot, mdot, ndot, hdot])
