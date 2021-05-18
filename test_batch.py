import run_batch as batch
import potential_models as models


def f(fssh):
    if fssh.x > 5 and fssh.v >= 0:
        return True
    elif fssh.x < -5 and fssh.v <= 0:
        return True
    return False


b = batch.Batch(models.Simple_Avoided_Crossing(), f, lambda _: _)
b.run(10, -10, 3)
b.generate_report("test")
b.output_csv("test")
