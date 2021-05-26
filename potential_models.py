import math
import numpy as np
import matplotlib.pyplot as plt


class Diabatic_Model:
    def __init__(self, num_states):
        self.num_states = num_states

    def get_adiabatic(self, x):
        v, ev = np.linalg.eig(self.V(x))
        d = {}
        for i in range(len(v)):
            d[v[i]] = ev[:, i]

        v_sorted = np.sort(v)
        ev_sorted = np.zeros((self.num_states, self.num_states))
        for i in range(len(v_sorted)):
            ev_sorted[:, i] = d[v[i]]

        return v_sorted, ev_sorted

    def get_adiabatic_energy(self, x):
        return self.get_adiabatic(x)[0]

    def get_wave_function(self, x):
        return self.get_adiabatic(x)[1]

    def get_d_adiabatic_energy(self, x, step=0.00001):
        v1 = self.get_adiabatic_energy(x + step)
        v0 = self.get_adiabatic_energy(x - step)
        return (v1 - v0)/(2*step)

    def get_d_wave_functions(self, x, step=0.00001):
        phi1 = self.get_wave_function(x + step)
        phi0 = self.get_wave_function(x - step)
        return (phi1 - phi0)/(2*step)


class Simple_Avoided_Crossing(Diabatic_Model):
    def __init__(self, A=0.01, B=1.6, C=0.005, D=1.0, discont=0):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.discont = discont
        self.num_states = 2

        super().__init__(self.num_states)

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


class Double_Avoided_Crossing(Diabatic_Model):
    def __init__(self, A=.1, B=.28, E0=.05, C=.015, D=.06):
        self.A = A
        self.B = B
        self.E0 = E0
        self.C = C
        self.D = D
        self.num_states = 2

        super().__init__(self.num_states)

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


class Extended_Coupling_With_Reflection(Diabatic_Model):
    def __init__(self, A=6e-4, B=.1, C=.9, discont=0):
        self.A = A
        self.B = B
        self.C = C
        self.discont = discont
        self.num_states = 2

        super().__init__(self.num_states)

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
    d21 = np.zeros(len(x_linspace))
    for i in range(len(x_linspace)):
        x = x_linspace[i]

        # Get adiabatic representation
        potential, ev = model.get_adiabatic(x)
        adiabatic_1[i] = potential[0]
        adiabatic_2[i] = potential[1]

        # Calcualte d12
        grad_phi = model.get_d_wave_functions(x)
        d12[i] = ev[:, 0]@grad_phi[:, 1]
        d21[i] = ev[:, 1]@grad_phi[:, 0]

    plt.plot(x_linspace, adiabatic_1)
    plt.plot(x_linspace, adiabatic_2)
    plt.plot(x_linspace, d12*coupling_scaling_factor)
   # plt.plot(x_linspace, d21*coupling_scaling_factor)
    plt.show()
