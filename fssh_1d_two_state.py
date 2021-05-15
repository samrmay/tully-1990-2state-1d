# 1d, 2 state version for learning/debugging. After this, extend to
# n-dimensions and m-states
import math
import numpy as np
import scipy.misc as misc
import scipy.integrate as integrate
import random as rand


class FSSH_1d:
    # Potential model has methods for returning the diabatic representation in
    # mtx form and the derivative of the
    # potential, both at position x.
    def __init__(self, potential_model, del_t, x0, v0, t0=0, m=2000,
                 coeff0=np.asarray([1, 0]), state0=0):
        self.potential_model = potential_model
        self.del_t = del_t
        self.x = x0
        self.v = v0
        self.t = t0
        self.m = m
        self.coeff = coeff0
        self.e_state = 0

        self.num_states = 2
        self.dim = 1

        self.HBAR = 1.055 * (10**-34)

    # Returns tuple (new x, new velocity). Uses 1-d kinematics and potential
    # gradient as acceleration
    def calc_trajectory(self, x0, m, v0, t1, e_state):
        a = -self.potential_model.dV(x0)[e_state, e_state]/m
        v1 = v0 + a*t1
        x1 = x0 + (v0*t1) + .5*a*(t1**2)

        return (x1, v1)

    # Returns tuple(energies, eigenvectors == electronic wave functions).
    # Retrieved from egienvalues/eigenfunctions of
    # diabatic representation
    def get_electronic_state(self, x):
        return np.linalg.eigh(self.potential_model.V(x))

    # Returns non-adiabatic coupling vectors given wave function vector
    def get_NACV(self, x, e_state):
        grad_state = np.zeros(self.num_states)

        # Calculate grad of electronic wave functions w.r.t R
        for i in range(self.num_states):
            def f(x):
                return self.get_electronic_state(x)[1][i]
            grad_state[i] = misc.derivative(f, x)

        # Nonadiabatic coupling vector -> dij = <phi_i | grad_R phi_j>
        d1 = [np.dot(e_state[0], grad_state[0]),
              np.dot(e_state[0], grad_state[1])]
        d2 = [np.dot(e_state[1], grad_state[0]),
              np.dot(e_state[1], grad_state[1])]

        return np.asarray([d1, d2])

    def get_density_mtx(self, x0, t0=0):
        # Function to return coefficients for a given t. R depends on t,
        # and rest of equation depends on R, so need to calculate R based on
        # current trajectory and t
        def f(t, c):
            x, dx = self.calc_trajectory(x0, self.m, self.v, t)
            e_state = self.get_electronic_state(x)
            nacvs = self.get_NACV(x0, e_state)
            V = self.potential_model.V(x)

            c1 = c[0]
            c2 = c[1]

            ih_bar = 1j*self.HBAR

            c1_dot = (1/(ih_bar)) * \
                ((c1*(V[0, 0] - ih_bar*np.dot(dx, nacvs[0, 0]))) +
                 (c2*(V[0, 1] - ih_bar*np.dot(dx, nacvs[0, 1]))))

            c2_dot = (1/(ih_bar)) * \
                ((c1*(V[1, 0] - ih_bar*np.dot(dx, nacvs[1, 0]))) +
                 (c2*(V[1, 1] - ih_bar*np.dot(dx, nacvs[1, 1]))))

            return [c1_dot, c2_dot]

        # Integrate equation from t=0, max t is delta_t for algorithm.
        # f takes algorithm's current velocity as starting velocity, so
        # shouldnt matter that starting t=0
        c1, c2 = integrate.RK45(f, t0, self.coeff, self.del_t)

        return np.asarray([[c1*(c1.conjugate()), c1*(c2.conjugate())],
                           [c2*(c1.conjugate()), c2*(c2.conjugate())]])

    # Determines whether a switch should be made given current state
    # (coefficients, position, density mtx, and coupling vectors). Returns bool
    # Since this version is specifically two_state, dont need to worry about
    # which state is being switched to
    def should_switch(self, x, density_mtx, nacv, V, e_state, del_t):
        b12 = (2/self.HBAR)*((density_mtx[0, 1].conjugate()*V[0, 1]).imag) - \
            2*((density_mtx[0, 1].conjugate()*np.dot(x, nacv[0, 1])).real)

        b21 = (2/self.HBAR)*((density_mtx[1, 0].conjugate()*V[1, 0]).imag) - \
            2*((density_mtx[1, 0].conjugate()*np.dot(x, nacv[1, 0])).real)
        b12 = min(0, b12)
        b21 = min(0, b21)

        g12 = (del_t*b21)/density_mtx[0, 0]
        g21 = (del_t*b12)/density_mtx[1, 1]
        delta = rand.random()

        if (e_state == 0 and g12 > delta):
            return True
        elif (e_state == 1 and g21 > delta):
            return True
        return False

    def run(self, max_step):
        # Variables should be initialized on class instantiation,
        # No need to init step (step 1)

        # Run for max number of steps or until stopping parameter is hit
        for _ in range(max_step):
            # Step 2a: Calculate x, v for small time step based on
            # current trajectory (based on current PES)
            x0 = self.x
            t0 = self.t
            v0 = self.v
            e_state0 = self.e_state
            t1 = t0 + self.del_t

            self.x, self.v = self.calc_trajectory(x0, self.m, v0, t1, e_state0)

            # Step 2b: calculate density mtx. along current trajectory
            d_mtx = self.get_density_mtx(x0)

            # Step 3: Determine if switch should occur by calculating g
            _, wave_functions = self.get_electronic_state(x0)
            nacv = self.get_NACV(x0, wave_functions)
            will_switch = self.should_switch(
                x0, d_mtx, nacv, self.potential_model.V(x0), wave_functions, self.del_t)


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
            V11 = -self.A*(1-(math.exp(-self.B*x)))

        V22 = -V11
        V12 = V21 = self.C*math.exp(-self.D*(x**2))

        return np.asarray([[V11, V12], [V21, V22]])

    def dV(self, x):
        if x > self.discont:
            dV11 = self.A*self.B*x*math.exp(-self.B*x)
        else:
            dV11 = -self.A*self.B*x*math.exp(-self.B*x)

        dV22 = -dV11
        dV12 = dV21 = -2*self.C*self.D*x*math.exp(-self.D*(x**2))

        return np.asarray([[dV11, dV12], [dV21, dV22]])
