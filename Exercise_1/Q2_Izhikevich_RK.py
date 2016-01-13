"""
Computational Neurodynamics
Exercise 1 Q2

http://www.izhikevich.org/publications/spikes.htm

(C) Murray Shanahan et al, 2015
"""

import numpy as np
import matplotlib.pyplot as plt

# Create time points
Tmin = 0
Tmax = 200  # Simulation time
dt = 0.01  # Step size
T = np.arange(Tmin, Tmax + dt, dt)

# Base current
I = 10

## Parameters of Izhikevich's model (regular spiking)
a = 0.02
b = 0.2
c = -65
d = 8

## Parameters of Izhikevich's model (fast spiking)
# a = 0.02
# b = 0.25
# c = -65
# d = 2

## Parameters of Izhikevich's model (bursting)
# a = 0.02
# b = 0.2
# c = -50
# d = 2

# v = np.zeros(len(T))
# u = np.zeros(len(T))

## Initial values
v0 = -65
u0 = -1

x = np.zeros((len(T), 2))  # tall vector of states [(v0,u0);(v1,u1)...]
x[0, :] = [v0, u0]


def xdot(x, I):
    vdot = 0.04 * x[0] ** 2 + 5 * x[0] + 140 - x[1] + I
    udot = a * (b * x[0] - x[1])
    return np.array([vdot, udot])


## SIMULATE
for t in xrange(len(T) - 1):
    # Update v and u according to Izhikevich's equations
    k1 = xdot(x[t], I)
    k2 = xdot(x[t] + 0.5 * dt * k1, I)
    k3 = xdot(x[t] + 0.5 * dt * k2, I)
    k4 = xdot(x[t] + dt * k3, I)

    x[t + 1] = x[t] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    # Reset the neuron if it has spiked
    if x[t + 1, 0] >= 30:
        x[t, 0] = 30  # Add a Dirac pulse for visualisation
        x[t + 1, 0] = c  # Reset to resting potential
        x[t + 1, 1] += d  # Update recovery variable

v = x[:, 0]
u = x[:, 1]

## Plot the membrane potential
plt.subplot(211)
plt.plot(T, v)
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential v (mV)')
plt.title('Izhikevich Neuron')

# Plot the reset variable
plt.subplot(212)
plt.plot(T, u)
plt.xlabel('Time (ms)')
plt.ylabel('Reset variable u')
plt.show()
