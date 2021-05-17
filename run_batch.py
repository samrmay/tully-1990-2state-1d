import fssh_1d_two_state as fssh


class Batch:
    def __init__(self, model, stopping_function, cat_function):
        self.model = model
        self.stopping_function = stopping_function
        self.cat_function = cat_function

        self.states = []

    def run(self, num_particles=10, start_x=-10, momentum=30, mass=2000, del_t=.5, max_iter=5000):
        for _ in range(num_particles):
            v = momentum/mass
            x = fssh.FSSH_1d(self.model, del_t, start_x, v, mass)
            x.run(max_iter, self.stopping_function)

            self.states.append((x.x, x.v, x.e_state))

    def report(self):
        for state in self.states:
            print(state)