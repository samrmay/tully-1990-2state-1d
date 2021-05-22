# 1d, 2 state version for learning/debugging. After this, extend to
# n-dimensions and m-states
import math
import numpy as np
import scipy.integrate as integrate
import random as rand


class FSSH_1d:
    # Potential model has methods for returning the diabatic representation in
    # mtx form and the derivative of the
    # potential, both at position x.
    def __init__(self, potential_model, del_t, x0, v0, m=2000, t0=0,
                 coeff0=np.asarray([1, 0], dtype=complex), state0=0):
        self.potential_model = potential_model
        self.del_t = del_t
        self.x = x0
        self.v = v0
        self.t = t0
        self.m = m
        self.coeff = coeff0
        self.e_state = state0
        self.i = 0

        self.num_states = 2
        self.dim = 1

        self.HBAR = 1

    # Returns tuple (new x, new velocity). Uses 1-d kinematics and adiabatic potential
    # gradient as acceleration
    def calc_trajectory(self, x0, m, v0, del_t, e_state):
        d_potential = self.potential_model.get_d_adiabatic_energy(x0)
        a = -d_potential[e_state]/m

        v1 = v0 + a*del_t
        x1 = x0 + (v0*del_t) + .5*a*(del_t**2)

        return (x1, v1)

    # Returns tuple(energies, eigenvectors == electronic wave functions).
    # Retrieved from egienvalues/eigenfunctions of
    # diabatic representation
    def get_electronic_state(self, x):
        return self.potential_model.get_adiabatic(x)

    # Returns non-adiabatic coupling vectors given wave function vector
    def get_NACV(self, x, e_state):
        grad_phi = self.potential_model.get_d_wave_functions(x)

        # Nonadiabatic coupling vector -> dij = <phi_i | grad_R phi_j>
        d1 = [e_state[:, 0]@grad_phi[:, 0],
              e_state[:, 0]@grad_phi[:, 1]]
        d2 = [e_state[:, 1]@grad_phi[:, 0],
              e_state[:, 1]@grad_phi[:, 1]]
        return np.asarray((d1, d2), dtype=complex)

    def get_density_mtx(self, x0, v0, e_state, t0=0):
        # Function to return coefficients for a given t. R depends on t,
        # and rest of equation depends on R, so need to calculate R based on
        # current trajectory and t
        def f(t, c):
            x, dx = self.calc_trajectory(x0, self.m, v0, t, e_state)
            _, e_functions = self.get_electronic_state(x)
            nacvs = self.get_NACV(x, e_functions)
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
        integrator = integrate.RK45(f, t0, self.coeff, self.del_t)

        while integrator.status == 'running':
            if integrator.step() == 'failed':
                raise BaseException("failed to solve density mtx.")

        if integrator.status == 'finished':
            c1, c2 = integrator.y
            self.coeff = c1, c2
        else:
            raise BaseException("failed to solve density mtx.")
        return np.asarray([[c1*(c1.conjugate()), c1*(c2.conjugate())],
                           [c2*(c1.conjugate()), c2*(c2.conjugate())]], dtype=complex)

    # Determines whether a switch should be made given current state
    # (coefficients, position, density mtx, and coupling vectors). Returns bool
    # Since this version is specifically two_state, dont need to worry about
    # which state is being switched to
    def should_switch(self, x, density_mtx, nacv, V, e_state, del_t):
        b12 = ((2/self.HBAR)*((density_mtx[0, 1].conjugate()*V[0, 1]).imag)) - \
            2*((density_mtx[0, 1].conjugate()*np.dot(x, nacv[0, 1])).real)

        b21 = (2/self.HBAR)*((density_mtx[1, 0].conjugate()*V[1, 0]).imag) - \
            2*((density_mtx[1, 0].conjugate()*np.dot(x, nacv[1, 0])).real)

        g12 = (del_t*b21)/density_mtx[0, 0] if density_mtx[0, 0] != 0 else 0
        g21 = (del_t*b12)/density_mtx[1, 1] if density_mtx[1, 1] != 0 else 0
        delta = rand.random()

        g12 = max(0, g12)
        g21 = max(0, g21)
        if (e_state == 0 and g12 > delta):
            return True
        elif (e_state == 1 and g21 > delta):
            return True
        return False

    def get_KE(self, m, v):
        return .5*m*(v**2)

    # Handles switch from old state to new state. Returns bool to denote
    # if state switch was successful (if energy cannot be conserved, state switch fails)
    # energy is from adiabatic representation
    def handle_switch(self, energy, m, v, nacv, old_state, new_state):
        KE = self.get_KE(m, v)
        new_V = energy[new_state]
        old_V = energy[old_state]
        diff = new_V - old_V
        # If no difference in potentials, no need to update velocity
        if diff == 0:
            self.e_state = new_state
            return True
        # Check if particle has enough KE to conserve energy. If not,
        # cancel state switch
        elif KE < diff:
            if self.debug:
                print("failed state switch: ", old_state,
                      "-/>", new_state, "@", self.x, " KE: ", KE, " deltaV: ", diff)
            return False
        else:
            # Since only 1d problem, dont have to worry about in
            # which direction to update velocity.
            direction = -1 if v < 0 else 1
            self.v = math.sqrt((v**2) - ((2/m)*diff))*direction
            self.e_state = new_state
            if self.debug:
                print("state switch: ", old_state,
                      "->", new_state, "@", self.x, " KE: ", KE, " deltaV: ", diff)
            return True

    def run(self, max_step, stopping_function, debug=False):
        # Variables should be initialized on class instantiation,
        self.debug = debug
        # Run for max number of steps or until stopping parameter is hit
        for i in range(max_step):
            # Step 2a: Calculate x, v for small time step based on
            # current trajectory (based on current PES)
            x0 = self.x
            t0 = self.t
            v0 = self.v
            m = self.m
            e_state0 = self.e_state
            t1 = t0 + self.del_t
            self.i += 1

            self.x, self.v = self.calc_trajectory(x0, m, v0, t1, e_state0)
            if (debug and i % 100 == 0):
                print(self.x, self.v, self.get_KE(m, self.v))

            # Step 2b: calculate density mtx. along current trajectory.
            # Method integrates along trajectory according to current state
            # until delta_t, so pass in x0, v0 as start conditions
            d_mtx = self.get_density_mtx(x0, v0, e_state0)

            # Step 3: Determine if switch should occur by calculating g.
            energy, wave_functions = self.get_electronic_state(self.x)
            nacv = self.get_NACV(self.x, wave_functions)
            V = self.potential_model.V(self.x)
            will_switch = self.should_switch(
                self.x, d_mtx, nacv, V, e_state0, self.del_t)

            # Step 4: switch if needed, update velocity if needed. Make sure
            # to pass in new velocity (not v0)
            if will_switch:
                self.handle_switch(energy, m, self.v, nacv, e_state0,
                                   1 if e_state0 == 0 else 0)

            # Step 5: Check stopping parameters using function passed in as argument.
            # Parameters of function described above
            if stopping_function(self):
                break
