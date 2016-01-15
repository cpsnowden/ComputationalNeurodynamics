"""
Computational Neurodynamics
Exercise 2

(C) Murray Shanahan et al, 2015
"""

import numpy as np
import numpy.random as rn
from HhNetwork import HhNetwork


def RobotConnect4L(Ns, Nm, Ni = None):
  """
  Construct four layers of Izhikevich neurons and connect them together.
  Layers 0 and 1 comprise sensory neurons, while layers 2 and 3 comprise
  motor neurons. Sensory neurons excite contralateral motor neurons causing
  seeking behaviour. Layers are heterogenous populations of Izhikevich
  neurons with slightly different parameter values.

  Inputs:
  Ns -- Number of neurons in sensory layers
  Nm -- Number of neurons in motor layers
  """

  F    = 50.0/np.sqrt(Ns)  # Scaling factor
  D    = 4                 # Conduction delay
  Dmax = 5                 # Maximum conduction delay

  if(Ni != None):
    net = HhNetwork([Ns, Ns, Nm, Nm, Ni, Ni], Dmax)
  else:
    net = HhNetwork([Ns, Ns, Nm, Nm], Dmax)


  gNa = 120.0
  gK = 36.0
  gL = 0.3
  ENa = 115.0
  EK = -12.0
  EL = 10.6
  C = 1.0

  # Layer 0 (Left sensory neurons)
  r = rn.rand(Ns)
  net.layer[0].Ns = Ns
  net.layer[0].gNa = np.ones(Ns) * gNa
  net.layer[0].gK = np.ones(Ns) * gK
  net.layer[0].gL = np.ones(Ns) * gL
  net.layer[0].ENa = np.ones(Ns) * ENa
  net.layer[0].EK = np.ones(Ns) * EK
  net.layer[0].EL = np.ones(Ns) * EL
  net.layer[0].C  = np.ones(Ns) * C

  # Layer 1 (Right sensory neurons)
  net.layer[1].Ns = Ns
  net.layer[1].gNa = np.ones(Ns) * gNa
  net.layer[1].gK = np.ones(Ns) * gK
  net.layer[1].gL = np.ones(Ns) * gL
  net.layer[1].ENa = np.ones(Ns) * ENa
  net.layer[1].EK = np.ones(Ns) * EK
  net.layer[1].EL = np.ones(Ns) * EL
  net.layer[1].C  = np.ones(Ns) * C
  
  # Layer 2 (Left motor neurons)
  net.layer[2].Ns = Ns
  net.layer[2].gNa = np.ones(Ns) * gNa
  net.layer[2].gK = np.ones(Ns) * gK
  net.layer[2].gL = np.ones(Ns) * gL
  net.layer[2].ENa = np.ones(Ns) * ENa
  net.layer[2].EK = np.ones(Ns) * EK
  net.layer[2].EL = np.ones(Ns) * EL
  net.layer[2].C  = np.ones(Ns) * C
  # Layer 3 (Right motor neurons)
  net.layer[3].Ns = Ns
  net.layer[3].gNa = np.ones(Ns) * gNa
  net.layer[3].gK = np.ones(Ns) * gK
  net.layer[3].gL = np.ones(Ns) * gL
  net.layer[3].ENa = np.ones(Ns) * ENa
  net.layer[3].EK = np.ones(Ns) * EK
  net.layer[3].EL = np.ones(Ns) * EL
  net.layer[3].C  = np.ones(Ns) * C

  if(Ni != None):

    # Layer 4 (Left inhibitory neurons)
    net.layer[4].Ns = Ns
    net.layer[4].gNa = np.ones(Ns) * gNa
    net.layer[4].gK = np.ones(Ns) * gK
    net.layer[4].gL = np.ones(Ns) * gL
    net.layer[4].ENa = np.ones(Ns) * ENa
    net.layer[4].EK = np.ones(Ns) * EK
    net.layer[4].EL = np.ones(Ns) * EL
    net.layer[4].C  = np.ones(Ns) * C
    # Layer 5 (Right inhibitory neurons)
    net.layer[5].Ns = Ns
    net.layer[5].gNa = np.ones(Ns) * gNa
    net.layer[5].gK = np.ones(Ns) * gK
    net.layer[5].gL = np.ones(Ns) * gL
    net.layer[5].ENa = np.ones(Ns) * ENa
    net.layer[5].EK = np.ones(Ns) * EK
    net.layer[5].EL = np.ones(Ns) * EL
    net.layer[5].C  = np.ones(Ns) * C

  
  # Connectivity matrix (synaptic weights)
  # layer[i].S[j] is the connectivity matrix from layer j to layer i
  # s[i,j] is the streght of the connection from neuron j to neuron i



  # Connect 0 to 3 and 1 to 2 for seeking behaviour
  net.layer[3].S[0]      = np.ones([Nm, Ns])
  net.layer[3].factor[0] = F
  net.layer[3].delay[0]  = D * np.ones([Nm, Ns], dtype=int)

  net.layer[2].S[1]      = np.ones([Nm, Ns])
  net.layer[2].factor[1] = F
  net.layer[2].delay[1]  = D * np.ones([Nm, Ns], dtype=int)

  if(Ni != None):

   #Connect Left Motor Neuron to Left Inhibitory Neuron
    net.layer[4].S[2]      = np.ones([Nm, Ns])
    net.layer[4].factor[2] = F
    net.layer[4].delay[2]  = D * np.ones([Nm, Ns], dtype=int)

    #Connect Right Motor Neuron to Right Inhibitory Neuron
    net.layer[5].S[3]      = np.ones([Nm, Ns])
    net.layer[5].factor[3] = F
    net.layer[5].delay[3]  = D * np.ones([Nm, Ns], dtype=int)

    inhib = -0.2
    #Connect Right Inhibitory Neuron to Left Inhibitory Neuron
    net.layer[4].S[5]      = inhib * np.ones([Nm, Ns])
    net.layer[4].factor[5] = F
    net.layer[4].delay[5]  = D * np.ones([Nm, Ns], dtype=int)
    #Connect Left Inhibitory Neuron to Right Inhibitory Neuron
    net.layer[5].S[4]      = inhib * np.ones([Nm, Ns])
    net.layer[5].factor[4] = F
    net.layer[5].delay[4]  = D * np.ones([Nm, Ns], dtype=int)

    #Connect Left Inhibitory Neuron to Right Motor Neuron
    net.layer[3].S[4]      = inhib * np.ones([Nm, Ns])
    net.layer[3].factor[4] = F
    net.layer[3].delay[4]  = D * np.ones([Nm, Ns], dtype=int)
    #Connect Right Inhibitory Neuron to Left Motor Neuron
    net.layer[2].S[5]      = inhib * np.ones([Nm, Ns])
    net.layer[2].factor[5] = F
    net.layer[2].delay[5]  = D * np.ones([Nm, Ns], dtype=int)

  return net

