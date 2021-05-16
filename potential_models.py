import math
import numpy as np
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

# Plots adiabatic potential from diabatic representation


def plot_adiabatic_potential(model, x0, x1):
    x_linspace = np.linspace(x0, x1)
    adiabatic_1 = np.zeros(len(x_linspace))
    adiabatic_2 = np.zeros(len(x_linspace))
    for i in range(len(x_linspace)):
        x = x_linspace[i]

        diabatic = model.V(x)
        adiabatic = np.diag(diabatic)
        adiabatic_1[i] = adiabatic[0]
        adiabatic_2[i] = adiabatic[1]

    plt.plot(x_linspace, adiabatic_1)
    plt.plot(x_linspace, adiabatic_2)
    plt.show()


plot_adiabatic_potential(Simple_Avoided_Crossing(), -10, 10)
