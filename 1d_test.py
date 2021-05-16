import fssh_1d_two_state as fssh_1d
import potential_models as models

model = models.Simple_Avoided_Crossing()

momentum = 5
m = 2000
v = momentum/m
alg = fssh_1d.FSSH_1d(model, .5, -10, v, m)


def f(fssh):
    if fssh.x > 5 and fssh.v >= 0:
        return True
    elif fssh.x < -5 and fssh.v <= 0:
        return True
    return False


alg.run(5000, f)
print(alg.x, alg.v, alg.e_state)
