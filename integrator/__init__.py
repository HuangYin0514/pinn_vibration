from .el import Euler, ImplicitEuler
from .rk import RK4, RK4_high_order

__factory = {
    "RK4": RK4,
    "RK4_high_order": RK4_high_order,
    "Euler": Euler,
    "ImplicitEuler": ImplicitEuler,
}


def ODEIntegrate(method, *args, **kwargs):
    if method not in __factory.keys():
        raise ValueError("solver '{}' is not implemented".format(method))
    results = __factory[method](*args, **kwargs).solve(*args, **kwargs)
    return results
