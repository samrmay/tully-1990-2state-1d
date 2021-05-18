import math
import numpy as np
import scipy.misc as misc
import matplotlib.pyplot as plt


class Simple_Avoided_Crossing:
    def __init__(self, A=0.01, B=1.6, C=0.005, D=1.0, discont=0):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.discont = discont

    def V(self, x):
        if x > self.discont:
            V11 = self.A*(1-(math.exp(-self.B*x)))
        else:
            V11 = -self.A*(1-(math.exp(self.B*x)))

        V22 = -V11
        V12 = V21 = self.C*math.exp(-self.D*(x**2))

        return np.asarray([[V11, V12], [V21, V22]])

    def dV(self, x):
        if x > self.discont:
            dV11 = self.A*self.B*x*math.exp(-self.B*x)
        else:
            dV11 = -self.A*self.B*x*math.exp(self.B*x)

        dV22 = -dV11
        dV12 = dV21 = -2*self.C*self.D*x*math.exp(-self.D*(x**2))

        return np.asarray([[dV11, dV12], [dV21, dV22]])


class Double_Avoided_Crossing:
    def __init__(self, A=.1, B=.28, E0=.05, C=.015, D=.06):
        self.A = A
        self.B = B
        self.E0 = E0
        self.C = C
        self.D = D

    def V(self, x):
        V11 = 0.0
        V22 = (-self.A*math.exp(-self.B*(x**2))) + self.E0
        V12 = V21 = self.C*math.exp(-self.D*(x**2))

        return np.asarray([[V11, V12], [V21, V22]])

    def dV(self, x):
        dV11 = 0
        dV22 = 2*self.A*self.B*x*math.exp(-self.B*(x**2))
        dV12 = dV21 = -2*self.C*self.D*x*math.exp(-self.D*(x**2))

        return np.asarray([[dV11, dV12], [dV21, dV22]])


class Extended_Coupling_With_Reflection:
    def __init__(self, A=6*(10**-4), B=.1, C=.9, discont=0):
        self.A = A
        self.B = B
        self.C = C
        self.discont = discont

    def V(self, x):
        V11 = self.A
        V22 = -self.A
        if x > self.discont:
            V12 = V21 = self.B*(2-math.exp(-self.C*x))
        else:
            V12 = V21 = self.B*math.exp(self.C*x)

        return np.asarray([[V11, V12], [V21, V22]])

    def dV(self, x):
        dV11 = 0
        dV22 = 0
        if x > self.discont:
            dV12 = dV21 = self.B*self.C*math.exp(-self.C*x)
        else:
            dV12 = dV21 = self.C*self.B*math.exp(self.C*x)

        return np.asarray([[dV11, dV12], [dV21, dV22]])

# Plots adiabatic potential from diabatic representation


def plot_adiabatic_potential(model, x0, x1, num_iter, coupling_scaling_factor):
    x_linspace = np.linspace(x0, x1, num_iter)

    # adiabatic potential vectors
    adiabatic_1 = np.zeros(len(x_linspace))
    adiabatic_2 = np.zeros(len(x_linspace))

    # d12 vector
    d12 = np.zeros(len(x_linspace))
    for i in range(len(x_linspace)):
        x = x_linspace[i]

        # Calculate potentials
        diabatic = model.V(x)
        lamda, ev = np.linalg.eig(diabatic)
        # Adiabatic potential is just eigenvalue
        adiabatic_1[i] = min(lamda)
        adiabatic_2[i] = max(lamda)
        # Calcualte d12
        grad_phi1 = np.zeros(2)

        def f(x1):
            return np.linalg.eig(model.V(x1))[1][1, 0]
        grad_phi1[0] = misc.derivative(f, x, .01, order=3)

        def f(x1):
            return np.linalg.eig(model.V(x1))[1][1, 1]
        grad_phi1[1] = misc.derivative(f, x, .01, order=3)

        d12[i] = -ev[0]@grad_phi1

    plt.plot(x_linspace, adiabatic_1)
    plt.plot(x_linspace, adiabatic_2)
    plt.plot(x_linspace, d12*coupling_scaling_factor)
    plt.show()
