"""
Hodgekin-Huxley Network

"""

import numpy as np
from SimulationMethods import eul, rk4
from NeuronModels import HodgekinHuxley

class HhNetwork:
    """

    """

    def __init__(self, _neuronsPerLayer, _Dmax):
        """
        Initialise network with given number of neurons

        Inputs:
        _neuronsPerLayer -- List with the number of neurons in each layer. A list
                            [N1, N2, ... Nk] will return a network with k layers
                            with the corresponding number of neurons in each.

        _Dmax            -- Maximum delay in all the synapses in the network. Any
                            longer delay will result in failing to deliver spikes.
        """

        self.Dmax = _Dmax
        self.Nlayers = len(_neuronsPerLayer)

        self.layer = {}

        for i, n in enumerate(_neuronsPerLayer):
            self.layer[i] = HhLayer(n)

    def Update(self, t):
        """
        Run simulation of the whole network for 1 millisecond and update the
        network's internal variables.

        Inputs:
        t -- Current timestep. Necessary to sort out the synaptic delays.
        """
        for lr in xrange(self.Nlayers):
            self.NeuronUpdate(lr, t)

    def NeuronUpdate(self, i, t):
        """
        Hodgekin-Huxley neuron update function. Update one layer for 1 millisecond
        using the Euler method.

        Inputs:
        i -- Number of layer to update
        t -- Current timestep. Necessary to sort out the synaptic delays.
        """

        # Euler method step size in ms
        dt = 0.02

        # Calculate current from incoming spikes
        for j in xrange(self.Nlayers):
            # LAYER I IS CONNECTED TO LAYER J IF THERE EXISTS A CONNECTIVITY MATIX S[J] I.E LAYER J --> LAYER I
            # If layer[i].S[j] exists then layer[i].factor[j] and
            # layer[i].delay[j] have to exist
            if j in self.layer[i].S:
                S = self.layer[i].S[j]  # target neuron->rows, source neuron->columns

                # Firings contains time and neuron idx of each spike.
                # [t, index of the neuron in the layer j]
                firings = self.layer[j].firings

                # Find incoming spikes taking delays into account
                delay = self.layer[i].delay[j]
                F = self.layer[i].factor[j]

                # Sum current from incoming spikes
                k = len(firings)
                while k > 0 and (firings[k - 1, 0] > (t - self.Dmax)):
                    idx = delay[:, firings[k - 1, 1]] == (t - firings[
                        k - 1, 0])  # which reveiving neurons have been waiting long enough as all elements of delay are the same, all idx will be true or false
                    self.layer[i].I[idx] += F * S[
                        idx, firings[k - 1, 1]]  # firings[k-1,1] is the index of neuron that has fired,
                    # idx the reveiving neurons which have been waiting long enough
                    k = k - 1

        # Update v and u using the HH model and RK4 method
        for k in xrange(int(1 / dt)):

            x = np.array([self.layer[i].v,
                          self.layer[i].m,
                          self.layer[i].n,
                          self.layer[i].h])

            fP = np.array([self.layer[i].I,
                           self.layer[i].gNa,
                           self.layer[i].gK,
                           self.layer[i].gL,
                           self.layer[i].ENa,
                           self.layer[i].EK,
                           self.layer[i].EL,
                           self.layer[i].C])

            self.layer[i].v, self.layer[i].m, self.layer[i].n, self.layer[i].h = rk4(x, dt, fP, HodgekinHuxley)

        #     print(i, k, "v:", self.layer[i].v)
        #     print(i, k, "m:", self.layer[i].m)
        #     print(i, k, "n:", self.layer[i].n)
        #     print(i, k, "h:", self.layer[i].h)
        #
        # print('\n')

        # Find index of neurons that have fired this millisecond
        fired = np.where(self.layer[i].v >= 50)[0]  # gives the index of the fired neurons i.e the column of the array

        if len(fired) > 0:
            for f in fired:
                # Add spikes into spike train
                if len(self.layer[i].firings) != 0:
                    self.layer[i].firings = np.vstack([self.layer[i].firings, [t, f]])
                else:
                    self.layer[i].firings = np.array([[t, f]])

        return



class HhLayer:
    """
    Layer of Hodgkin-Huxley neurons to be used inside an IzNetwork.
    """

    def __init__(self, n):
        """
        Initialise layer with empty vectors.

        Inputs:
        n -- Number of neurons in the layer
        """

        self.N = n
        self.gNa = np.zeros(n)
        self.gK = np.zeros(n)
        self.gL = np.zeros(n)
        self.ENa = np.zeros(n)
        self.EK = np.zeros(n)
        self.EL = np.zeros(n)
        self.C = np.zeros(n)

        self.S = {}
        self.delay = {}
        self.factor = {}





