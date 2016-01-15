"""
Simulation Methods

"""

def rk4(x_prev, dt, param, diff):
    """
    Implements Runge-Kutta 4 Method to progress the next state vector

    :param x_prev: Previous State Vector
    :type x_prev: np.array
    :param dt: Time step
    :type dt: double
    :param param: Differential Equation Parameters
    :type param: np.array
    :param diff: Function that returns the derivative of the state vector
    :type diff: Function taking state x and param
    :return: Next State Vector
    :rtype: np.array
    """

    k1 = diff(x_prev, param)
    k2 = diff(x_prev + 0.5 * dt * k1, param)
    k3 = diff(x_prev + 0.5 * dt * k2, param)
    k4 = diff(x_prev + dt * k3, param)

    return x_prev + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

def eul(x_prev, dt, param, diff):

    """
    Implements Euler Method to progress the next state vector

    :param x_prev: Previous State Vector
    :type x_prev: np.array
    :param dt: Time step
    :type dt: double
    :param param: Differential Equation Parameters
    :type param: np.array
    :param diff: Function that returns the derivative of the state vector
    :type diff: Function taking state x and param
    :return: Next State Vector
    :rtype:  np.array
    """
    return x_prev + dt * diff(x_prev, param)