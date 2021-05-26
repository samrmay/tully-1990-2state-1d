import run_batch as batch
import potential_models as models


def f(fssh):
    if fssh.x > 6 and fssh.v >= 0:
        return True
    elif fssh.x < -6 and fssh.v <= 0:
        return True
    return False


dir_ = "results/052521_test/"

print("30k")
b1 = batch.Batch(models.Simple_Avoided_Crossing(), f, lambda _: _)
b1.run(100, -10, 30, 10000, False, 2000, 10)
b1.generate_report(dir_ + "simple_30k_3")
b1.output_csv(dir_ + "simple_30k_3")

print("30k")
b2 = batch.Batch(models.Simple_Avoided_Crossing(), f, lambda _: _)
b2.run(200, -10, 30, 10000, False, 2000, 10)
b2.generate_report(dir_ + "simple_30k")
b2.output_csv(dir_ + "simple_30k")

print("27.5k")
b3 = batch.Batch(models.Simple_Avoided_Crossing(), f, lambda _: _)
b3.run(200, -10, 27.5, 10000, False, 2000, 10)
b3.generate_report(dir_ + "simple_27dot5k")
b3.output_csv(dir_ + "simple_27dot5k")

print("25k")
b4 = batch.Batch(models.Simple_Avoided_Crossing(), f, lambda _: _)
b4.run(200, -10, 25, 10000, False, 2000, 10)
b4.generate_report(dir_ + "simple_25k")
b4.output_csv(dir_ + "simple_25k")

print("22.5k")
b5 = batch.Batch(models.Simple_Avoided_Crossing(), f, lambda _: _)
b5.run(200, -10, 22.5, 10000, False, 2000, 10)
b5.generate_report(dir_ + "simple_22dot5k")
b5.output_csv(dir_ + "simple_22dot5k")

print("20k")
b6 = batch.Batch(models.Simple_Avoided_Crossing(), f, lambda _: _)
b6.run(200, -10, 20, 10000, False, 2000, 10)
b6.generate_report(dir_ + "simple_20k")
b6.output_csv(dir_ + "simple_20k")

print("15k")
b7 = batch.Batch(models.Simple_Avoided_Crossing(), f, lambda _: _)
b7.run(200, -10, 15, 10000, False, 2000, 15)
b7.generate_report(dir_ + "simple_15k")
b7.output_csv(dir_ + "simple_15k")

print("10k")
b8 = batch.Batch(models.Simple_Avoided_Crossing(), f, lambda _: _)
b8.run(200, -10, 10, 10000, False, 2000, 15)
b8.generate_report(dir_ + "simple_10k")
b8.output_csv(dir_ + "simple_10k")
