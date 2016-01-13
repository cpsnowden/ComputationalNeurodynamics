"""
Computational Neurodynamics
Comparison

(C) Murray Shanahan et al, 2015
"""

import numpy as np
import matplotlib.pyplot as plt
from Q4_RK_HH import rk4
from Q3_Euler_HH import eul

# Create time points
Tmin = 0
Tmax = 100  # Simulation time
dt = 0.08  # Step size
T = np.arange(Tmin, Tmax + dt, dt)

# Base current
I = 10.0
gNa = 120.0
gK = 36.0
gL = 0.3
ENa = 115.0
EK = -12.0
EL = 10.6
C = 1.0

fP = np.array([gNa, gK, gL, ENa, EK, EL, C])

x_eul = np.zeros((len(T),4))

## Initial values
v0= -10
m0 = 0.0
n0 = 0.0
h0 = 0.0

x_eul[0] = [v0, m0, n0, h0]
x_rk4 = x_eul.copy()

## SIMULATE
for t in xrange(len(T) - 1):
    x_rk4[t+1] = rk4(x_rk4[t], I, dt, fP)
    x_eul[t+1] = eul(x_eul[t], I, dt, fP)


v_rk4 = x_rk4[:,0]
v_eul = x_eul[:,0]

plt.plot(T, v_eul, 'b', label='Euler')
plt.plot(T, v_rk4, 'g', label='RK4')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential v (mV)')
plt.title('Hodgkin-Huxley Neuron')
plt.legend(loc=0)
plt.show()
plt.ion()