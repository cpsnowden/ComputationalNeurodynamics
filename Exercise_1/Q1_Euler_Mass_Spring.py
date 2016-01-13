"""
Computational Neurodynamics
Exercise 1 Q1

(C) Murray Shanahan et al, 2015
"""

import numpy as np
import matplotlib.pyplot as plt

dt = 0.001  # Step size for exact solution

# Spring coefficients
m = 1
c = 0.1
k = 1

# Create time points
Tmin = 0
Tmax = 100
T = np.arange(Tmin, Tmax + dt, dt)
y = np.zeros(len(T))
dy = np.zeros(len(T))

# Approximated solution with small integration Step
# Initial value
y[0] = 1
dy[0] = 0

for t in xrange(1, len(T)):
    y[t] = y[t - 1] + dt * dy[t - 1]
    d2y = (-1 / m) * (c * dy[t - 1] + k * y[t - 1])
    dy[t] = dy[t - 1] + dt * d2y

# Plot the results
plt.plot(T, y, 'b', label='Euler Solution to Spring/Damper System')
plt.xlabel('t')
plt.ylabel('y')
plt.legend(loc=0)
plt.show()
