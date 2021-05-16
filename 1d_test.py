import fssh_1d_two_state as fssh_1d

model = fssh_1d.Simple_Avoided_Crossing()

momentum = 35
m = 2000
v = momentum/m
alg = fssh_1d.FSSH_1d(model, .5, -10, v, m)
alg.run(2000)
