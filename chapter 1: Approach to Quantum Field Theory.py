from sympy import symbols, Sum, Indexed, lambdify, Array
from sympy.physics.quantum import Ket

import numpy as np

n, nu = symbols("n nu")
N = Sum(Indexed('n', nu), (nu, 0, 3))
f = lambdify(n, N)
b = []
n_nu = []
for nu in range(1, 5):
    arr1 = np.array([nu])
    b = np.concatenate((b, arr1))
    arr2 = Indexed('n', nu)
    n_nu = np.concatenate((n_nu, Array(arr2)))

print(f(b))

ket_n_nu = Ket(n_nu)  # Eqn (1.1)
print(ket_n_nu)
