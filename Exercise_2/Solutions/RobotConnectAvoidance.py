"""
Computational Neurodynamics
Exercise 2

(C) Murray Shanahan et al, 2015
"""

import numpy as np
import numpy.random as rn
from Exercise_2.IzNetwork import IzNetwork


def RobotConnectAvoidance(Ns, Nm):
  """
  Construct four layers of Izhikevich neurons and connect them together, just
  as RobotConnect4L. Layers 0 and 1 comprise sensory neurons, while layers 2
  and 3 comprise motor neurons. In this case, sensory neurons excite
  ipsilateral motor neurons causing avoidance behaviour.

  Inputs:
  Ns -- Number of neurons in sensory layers
  Nm -- Number of neurons in motor layers
  """

  F    = 50.0/np.sqrt(Ns)  # Scaling factor
  D    = 4                 # Conduction delay
  Dmax = 5                 # Maximum conduction delay

  net = IzNetwork([Ns, Ns, Nm, Nm], Dmax)

  # Layer 0 (Left sensory neurons)
  r = rn.rand(Ns)
  net.layer[0].N = Ns
  net.layer[0].a = 0.02 * np.ones(Ns)
  net.layer[0].b = 0.20 * np.ones(Ns)
  net.layer[0].c = -65 + 15*(r**2)
  net.layer[0].d = 8 - 6*(r**2)

  # Layer 1 (Right sensory neurons)
  r = rn.rand(Ns)
  net.layer[1].N = Ns
  net.layer[1].a = 0.02 * np.ones(Ns)
  net.layer[1].b = 0.20 * np.ones(Ns)
  net.layer[1].c = -65 + 15*(r**2)
  net.layer[1].d = 8 - 6*(r**2)

  # Layer 2 (Left motor neurons)
  r = rn.rand(Nm)
  net.layer[2].N = Nm
  net.layer[2].a = 0.02 * np.ones(Nm)
  net.layer[2].b = 0.20 * np.ones(Nm)
  net.layer[2].c = -65 + 15*(r**2)
  net.layer[2].d = 8 - 6*(r**2)

  # Layer 3 (Right motor neurons)
  r = rn.rand(Nm)
  net.layer[3].N = Nm
  net.layer[3].a = 0.02 * np.ones(Nm)
  net.layer[3].b = 0.20 * np.ones(Nm)
  net.layer[3].c = -65 + 15*(r**2)
  net.layer[3].d = 8 - 6*(r**2)

  # Connectivity matrix (synaptic weights)
  # layer[i].S[j] is the connectivity matrix from layer j to layer i
  # s[i,j] is the streght of the connection from neuron j to neuron i

  # Connect 0 to 2 and 1 to 3 for seeking behaviour
  net.layer[2].S[0]      = 5*np.ones([Nm, Ns])
  net.layer[2].factor[0] = F
  net.layer[2].delay[0]  = D * np.ones([Nm, Ns], dtype=int)

  net.layer[3].S[1]      = 5*np.ones([Nm, Ns])
  net.layer[3].factor[1] = F
  net.layer[3].delay[1]  = D * np.ones([Nm, Ns], dtype=int)

  return net

