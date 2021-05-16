import fssh_1d_two_state as fssh_1d

model = fssh_1d.Simple_Avoided_Crossing()
alg = fssh_1d.FSSH_1d(model, .1, -10, 1)
alg.run(1)
