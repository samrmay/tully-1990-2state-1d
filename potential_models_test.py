import potential_models as models


def simple():
    models.plot_adiabatic_potential(
        models.Simple_Avoided_Crossing(), -10, 10, 200, 1/50)


def double():
    models.plot_adiabatic_potential(
        models.Double_Avoided_Crossing(), -10, 10, 200, 1/12)


def extended():
    models.plot_adiabatic_potential(
        models.Extended_Coupling_With_Reflection(), -10, 10, 200, 1)


simple()
double()
extended()
