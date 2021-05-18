from os import write
import fssh_1d_two_state as fssh
from datetime import date
import time


class Batch:
    def __init__(self, model, stopping_function, cat_function):
        self.model = model
        self.stopping_function = stopping_function
        self.cat_function = cat_function

        self.batch_state = "initiated"
        self.batch_error = None

        self.start_time = time.time()
        self.end_time = self.start_time

        self.states = []

    def run(self, num_particles=10, start_x=-10, momentum=30, mass=2000, del_t=.5, max_iter=5000):
        self.k = momentum
        self.m = mass
        self.del_t = del_t
        self.max_iter = max_iter
        self.num_particles = 10
        self.start_x = start_x

        try:
            self.batch_state = "finished"
            for i in range(num_particles):
                print(i, "/", num_particles)
                v = momentum/mass
                x = fssh.FSSH_1d(self.model, del_t, start_x, v, mass)
                x.run(max_iter, self.stopping_function)

                self.states.append((x.x, x.v, x.e_state))
        except Exception as e:
            self.batch_state = "failed"
            self.batch_error = e
        finally:
            self.end_time = time.time()

    def generate_report(self, outfile):
        outfile += ".txt"
        with open(outfile, 'w') as f:
            lines = []
            lines.append(date.today().isoformat())
            lines.append(f"Job state: {self.batch_state}")
            lines.append(f"Potential model: {type(self.model).__name__}")
            lines.append(
                f"Job time: {self.end_time - self.start_time} seconds")

            if self.batch_state == "failed":
                lines.append("Job failed...")
                lines.append(self.batch_error)
                f.writelines(lines)
            elif self.batch_state == "initiated":
                lines.append("Job has not been run...")
                f.writelines(lines)
            else:
                lines.append(("-"*10) + "Job parameters" + ("-"*10))
                lines.append(f"Num particles: {self.num_particles}")
                lines.append(f"Max iter: {self.max_iter}")
                lines.append(f"Time step: {self.del_t}")
                lines.append(f"Particle momentum: {self.k}")
                lines.append(f"Start position: {self.start_x}")
                lines.append(f"Particle mass: {self.m}")

                lines.append(("-"*10) + "Job results" + ("-"*10))
                f.writelines(lines)
                self.enumerate_states(f)

    def enumerate_states(self, f):
        avg_pos = 0
        avg_state = 0
        avg_v = 0
        lines = []

        lines.append(("-"*10) + "Particle results" + ("-"*10))
        for i in range(len(self.states)):
            state = self.states[i]
            lines.append(str(i))
            lines.append(f"position: {state.x}")
            lines.append(f"velocity: {state.v}")
            lines.append(f"electronic state: {state.e_state}")
            lines.append(f"end_time: {state.t}")
            lines.append(
                f"end electronic coefficients: {state.coeff[0]}, {state.coeff[1]}")

            avg_pos += state.x
            avg_v += state.v
            avg_state += state.e_state

        avg_pos /= self.num_particles
        avg_v /= self.num_particles
        avg_state /= self.num_particles

        f.write(f"Avg position: {avg_pos}\n")
        f.write(f"Avg velocity: {avg_v}\n")
        f.write(f"Avg electronic state: {avg_state}\n")
        f.writelines(lines)
        return

    def output_csv(self, outfile):
        outfile += ".csv"
        with open(outfile, 'w') as f:
            f.write("position,velocity,electronic_state\n")
            for state in self.states:
                f.write(f"{state.x},{state.v},{state.e_state}\n")
