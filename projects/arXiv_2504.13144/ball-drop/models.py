#!/usr/bin/env python3
import numpy as np
import scipy as sp
from scipy.integrate import odeint

#==============================================================================

class BallDrop():
    """
    Initialize the model for ball drop experiment.
    Parameters:
        m (float): Mass of the ball
        D (float): Diameter of the ball
        beta (float): Coefficient of linear drag
        gamma (float): Coefficient of quadratic drag
    """
    def __init__(self, m=1.0, D=0.1, beta=0.01, gamma=40.0):
        self.m = m
        self.b = beta*D
        self.c = gamma * D**2

    def _equations(self, y, time, g):
        # function for defining derivatives of velocity and height
        h, v = y  # y is a list containing the dependent variables [v, h]

        dhdt = -v 
        dvdt = g - (self.b * v / self.m) - (self.c * abs(v) * v / self.m)  
        return [dhdt, dvdt]
        
    def height_velocity_accln(self, t=[1.0], g=9.8, v0=0.0, h0=60.0, t0=0.0):
        """
        Function to calculate the velocity, height, and acceleration of the ball at
        times t given initial velocity v0 and initial height h0 and g.

        Parameters:
            t (float or numpy 1darray): Point at which velocity has to be evaluated
            t0 (float): initial time
            h0 (float): initial height of the ball
            v0 (float): initial velocity of the ball
            g (float): acceleration due to gravity
 
        Returns:
             height, velocity, and acceleration of ball at times t (numpy ndarray)
        """ 
        t0 = np.atleast_1d(t0)
        t = np.atleast_1d(t)
        
        time = np.concatenate(([t0, t]))  # include initial time t0 in the time list
        ic = [h0, v0]
        sol = odeint(self._equations, ic, time, args=(g,))

        h = sol[:,0]  # Height is in the first column
        v = sol[:,1]  # Velocity is in the second column
        # Compute acceleration at each time step
        a = g - (self.b * v / self.m) - (self.c * abs(v) * v / self.m)

        return np.column_stack((h[1:], v[1:], a[1:]))


#==============================================================================







