# Quantum Functions

# Anticommutator
# Clebsch-Gordan Coefficient
# Commutator

from sympy import symbols
from sympy.physics.quantum import AntiCommutator
from sympy.physics.quantum import Operator, Dagger

x, y = symbols('x,y')
A = Operator('A')
B = Operator('B')
ac = AntiCommutator(A, B);
print(ac)
print(ac.doit())
print(AntiCommutator(3 * x * A, x * y * B))

# Adjoint operator
print(Dagger(AntiCommutator(A, B)))

# Clebsch-Gordan Coefficient

from sympy.physics.quantum.cg import CG
from sympy import S

cg = CG(S(3) / 2, S(3) / 2, S(1) / 2, -S(1) / 2, 1, 1)
print(cg)
print(cg.doit())

from sympy.physics.quantum.cg import Wigner3j

w3j = Wigner3j(6, 0, 4, 0, 2, 0)
print(w3j)
Wigner3j(6, 0, 4, 0, 2, 0)
print(w3j.doit())

from sympy.physics.quantum.cg import CG, cg_simp

a = CG(1, 1, 0, 0, 1, 1)
b = CG(1, 0, 0, 0, 1, 0)
c = CG(1, -1, 0, 0, 1, -1)
print(cg_simp(a + b + c))

# Commutator
from sympy.physics.quantum import Commutator, Dagger, Operator
from sympy.abc import x, y

A = Operator('A')
B = Operator('B')
C = Operator('C')

comm = Commutator(A, B)
print(comm)
print(comm.doit())

comm = Commutator(B, A)
print(comm)

print(Commutator(3 * x * A, x * y * B))
print(Commutator(A + B, C).expand(commutator=True))
print(Commutator)

print(Commutator(A + B, C).expand(commutator=True))
print(Commutator(A, B + C).expand(commutator=True))
print(Commutator(A * B, C).expand(commutator=True))
print(Commutator(A, B * C).expand(commutator=True))

print(Dagger(Commutator(A, B)))

# Constants
from sympy.physics.quantum.constants import hbar

print(hbar.evalf())

# Dagger
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.state import Ket, Bra
from sympy.physics.quantum.operator import Operator

print(Dagger(Ket('psi')))
print(Dagger(Bra('phi')))
print(Dagger(Operator('A')))

# Inner and outer product
from sympy.physics.quantum import InnerProduct, OuterProduct

print(Dagger(InnerProduct(Bra('a'), Ket('b'))))
print(Dagger(OuterProduct(Ket('a'), Bra('b'))))

# Powers, sum and product
A = Operator('A')
B = Operator('B')
print(Dagger(A * B))
print(Dagger(A + B))
print(Dagger(A ** 2))

# Complex numbers and matrices
from sympy import Matrix, I

m = Matrix([[1, I], [2, I]])
print(m)
print(Dagger(m))

# Inner product
from sympy.physics.quantum import Bra, Ket, InnerProduct

b = Bra('b')
print(b)
k = Ket('k')
print(k)
ip = b * k
print(ip)
print(ip.bra)
print(ip.ket)

print(b * k)
print(k * b * k * b)

print(k * (b * k) * b)

# Tensor product
from sympy import I, Matrix, symbols
from sympy.physics.quantum import TensorProduct

m1 = Matrix([[1, 2], [3, 4]])
m2 = Matrix([[1, 0], [0, 1]])
print(TensorProduct(m1, m2))

print(TensorProduct(m2, m1))

from sympy import Symbol

A = Symbol('A', commutative=False)
B = Symbol('B', commutative=False)
tp = TensorProduct(A, B)
print(tp)

# A = Symbol('A', commutative=True)
# B = Symbol('B', commutative=True)
# tp = TensorProduct(A, B)
# print(tp)

from sympy.physics.quantum import Dagger

print(Dagger(tp))

C = Symbol('C', commutative=False)
tp = TensorProduct(A + B, C)
print(tp)
print(tp.expand(tensorproduct=True))

from sympy.physics.quantum import tensor_product_simp
from sympy.physics.quantum import TensorProduct
from sympy import Symbol

A = Symbol('A', commutative=False)
B = Symbol('B', commutative=False)
C = Symbol('C', commutative=False)
D = Symbol('D', commutative=False)

e = TensorProduct(A, B) * TensorProduct(C, D)
print(e)
print(tensor_product_simp(e ** 2))

# States and Operators

# Cartesian Operators and States

# Hilbert Space
from sympy import symbols
from sympy.physics.quantum.hilbert import ComplexSpace

c1 = ComplexSpace(2)
print(c1)
print(c1.dimension)

n = symbols(('n'))
c2 = ComplexSpace(n)
print(c2)
print(c2.dimension)

from sympy.physics.quantum.hilbert import ComplexSpace, FockSpace
from sympy import symbols

c = ComplexSpace(2)
f = FockSpace()
hs = c + f
print(hs)
print(hs.dimension)
print(list(hs.spaces))

from sympy.physics.quantum.hilbert import FockSpace

hs = FockSpace()
print(hs)
print(hs.dimension)

from sympy.physics.quantum.hilbert import HilbertSpace

hs = HilbertSpace()
print(hs)

from sympy import Interval, oo
from sympy.physics.quantum.hilbert import L2

hs = L2(Interval(0, oo))
print(hs)

print(hs.dimension)
print(hs.interval)

from sympy.physics.quantum.hilbert import ComplexSpace, FockSpace
from sympy import symbols

n = symbols('n')
c = ComplexSpace(2)
hs = c ** n
print(hs)

print(hs.dimension)

c = ComplexSpace(2)
print(c * c)
f = FockSpace()
print(c * f * f)

from sympy.physics.quantum.hilbert import ComplexSpace, FockSpace
from sympy import symbols

c = ComplexSpace(2)
f = FockSpace()
hs = c * f
print(hs)
print(hs.dimension)
print(hs.spaces)

c = ComplexSpace(2)
n = symbols('n')
c2 = ComplexSpace(n)
hs = c1 * c2
print(hs)
print(hs.dimension)

# Operator
from sympy import Derivative, Function, Symbol
from sympy.physics.quantum.operator import DifferentialOperator
from sympy.physics.quantum.state import Wavefunction
from sympy.physics.quantum.qapply import qapply

f = Function('f')
x = Symbol('x')
d = DifferentialOperator(1 / x * Derivative(f(x), x), f(x))
w = Wavefunction(x ** 2, x)
print(d.function)
print(d.variables)
print("qapply:", qapply(d * w))

from sympy.physics.quantum.operator import DifferentialOperator
from sympy import Function, Symbol, Derivative

x = Symbol('x')
f = Function('f')
d = DifferentialOperator(Derivative(f(x), x), f(x))
print(d.expr)

y = Symbol('y')
d = DifferentialOperator(Derivative(f(x, y), x) + Derivative(f(x, y), y), f(x, y))
print(d.expr)

from sympy.physics.quantum.operator import DifferentialOperator
from sympy import Function, Symbol, Derivative

x = Symbol('x')
f = Function('f')
d = DifferentialOperator(Derivative(f(x), x), f(x))
print(d.function)

y = Symbol('y')
d = DifferentialOperator(Derivative(f(x, y), x) + Derivative(f(x, y), y), f(x, y))
print(d.function)

from sympy.physics.quantum.operator import DifferentialOperator
from sympy import Symbol, Function, Derivative

x = Symbol('x')
f = Function('f')
d = DifferentialOperator(1 / x * Derivative(f(x), x), f(x))
print(d.variables)

y = Symbol('y')
d = DifferentialOperator(Derivative(f(x, y), x) + Derivative(f(x, y), y), f(x, y))
print(d.variables)

from sympy.physics.quantum import Dagger, HermitianOperator

H = HermitianOperator('H')
print(Dagger(H))

from sympy.physics.quantum import IdentityOperator

print(IdentityOperator())

from sympy.physics.quantum import Operator
from sympy import symbols, I

A = Operator('A')
print(A)
print(A.hilbert_space)
print(A.label)
print(A.is_commutative)

B = Operator('B')
C = 2 * A * A + I * B
print(C)

print(A.is_commutative)
print(B.is_commutative)
print(A * B == B * A)

e = (A + B) ** 3
print(e.expand())

print(A.inv())
print(A * A.inv())

from sympy.physics.quantum import Ket, Bra, OuterProduct, Dagger
from sympy.physics.quantum import Operator

k = Ket('k')
b = Bra('b')
op = OuterProduct(k, b)
print(op)
print(op.hilbert_space)
print(op.ket)
print(op.bra)
print(Dagger(op))

print(k * b)

A = Operator('A')
print(A * k * b)

print(A * (k * b))

from sympy.physics.quantum import Dagger, UnitaryOperator

U = UnitaryOperator('U')
print(U * Dagger(U))

from sympy.physics.quantum.cartesian import XOp, PxOp
from sympy.physics.quantum.operatorset import operators_to_state
from sympy.physics.quantum.operator import Operator

print(operators_to_state(XOp))
print(operators_to_state(XOp()))
print(operators_to_state((PxOp)))
print(operators_to_state(PxOp()))
print(operators_to_state(Operator))
print(operators_to_state(Operator()))

from sympy.physics.quantum.cartesian import XKet, PxKet, XBra, PxBra
from sympy.physics.quantum.operatorset import state_to_operators
from sympy.physics.quantum.state import Ket, Bra

print(state_to_operators((XKet)))
print(state_to_operators((XKet())))
print(state_to_operators((PxKet)))
print(state_to_operators((PxKet())))
print(state_to_operators((PxBra)))
print(state_to_operators(XBra))
print(state_to_operators(Ket))
print(state_to_operators(Bra))

from sympy.physics.quantum import qapply, Ket, Bra

b = Bra('b')
k = Ket('k')
A = k * b
print(A)
print(qapply(A * b.dual / (b * b.dual)))
print(qapply(k.dual * A / (k.dual * k), dagger=True))
print(qapply(k.dual * A) / (k.dual * k))

# Represent
from sympy.physics.quantum.cartesian import XBra, XKet
from sympy.physics.quantum.represent import enumerate_states

test = XKet('foo')
print(enumerate_states(test, 1, 3))
test2 = XBra('bar')
print(enumerate_states(test2, [4, 5, 10]))

from sympy.physics.quantum.represent import get_basis
from sympy.physics.quantum.cartesian import XOp, XKet, PxOp, PxKet

x = XKet()
X = XOp()
print(get_basis(x))
print(get_basis(X))
print(get_basis(x, basis=PxOp()))
print(get_basis(x, basis=PxKet))

from sympy import symbols, DiracDelta
from sympy.physics.quantum.represent import integrate_result
from sympy.physics.quantum.cartesian import XOp, XKet

x_ket = XKet()
X_op = XOp()
x, x_1, x_2 = symbols('x, x_1, x_2')
print(integrate_result(X_op * x_ket, x * DiracDelta(x - x_1) * DiracDelta(x_1 - x_2)))
print(integrate_result(X_op * x_ket, x * DiracDelta(x - x_1) * DiracDelta(x_1 - x_2), unities=[1]))

from sympy.physics.quantum.represent import rep_innerproduct
from sympy.physics.quantum.cartesian import XOp, XKet, PxOp, PxKet

print(rep_innerproduct(XKet()))
print(rep_innerproduct(XKet(), basis=PxOp()))

from sympy.physics.quantum import Operator, represent, Ket
from sympy import Matrix


class SzUpKet(Ket):
    def _represent_SzOp(self, basis, **options):
        return Matrix([1, 0])


class SzOp(Operator):
    pass


sz = SzOp('Sz')
up = SzUpKet('up')
print(represent(up, basis=sz))

from sympy.physics.quantum.cartesian import XOp, XKet, XBra

X = XOp()
x = XKet()
y = XBra('y')
print(represent(x * X))
print(represent(X * x))
print(represent(X * x * y))

# Spin
from sympy.physics.quantum.spin import JzKet, JxKet
from sympy import symbols

print(JzKet(1, 0))
j, m = symbols('j m')
print(JzKet(j, m))

from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.spin import Jx, Jz

print(represent(JzKet(1, -1), basis=Jx))

from sympy.physics.quantum.innerproduct import InnerProduct
from sympy.physics.quantum.spin import JxBra

i = InnerProduct(JxBra(1, 1), JzKet(1, 1))
print(i)
print(i.doit())

from sympy.physics.quantum.tensorproduct import TensorProduct

j1, m1, j2, m2 = symbols('j1 m1 j2 m2')
print(TensorProduct(JzKet(1, 0), JzKet(1, 1)))
print(TensorProduct(JzKet(j1, m1), JzKet(j2, m2)))
print(TensorProduct(JzKet(1, 1), JxKet(1, 1)).rewrite('Jz'))

print(represent(TensorProduct(JzKet(1, 0), JzKet(1, 1))))
print(represent(TensorProduct(JzKet(1, 1), JxKet(1, 1)), basis=Jz))

from sympy.physics.quantum.spin import JzKetCoupled
from sympy import symbols

print(JzKetCoupled(1, 0, (1, 1)))
j, m, j1, j2 = symbols('j m j1 j2')
print(JzKetCoupled(j, m, (j1, j2)))

print(JzKetCoupled(2, 1, (1, 1, 1)))
print(JzKetCoupled(2, 1, (1, 1, 1), ((1, 2, 2), (1, 3, 2))))
print(JzKetCoupled(2, 1, (1, 1, 1), ((2, 3, 1), (1, 2, 2))))

print(JzKetCoupled(1, 1, (1, 1)).rewrite("Jx"))

print(JzKetCoupled(1, 0, (1, 1)).rewrite('Jz', coupled=False))

from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.spin import Jx
from sympy import S

print(represent(JzKetCoupled(1, -1, (S(1) / 2, S(1) / 2)), basis=Jx))

from sympy import pi
from sympy.physics.quantum.spin import Rotation

print(Rotation(pi, 0, pi / 2))

from sympy import symbols

a, b, c = symbols('a b c')
print(Rotation(a, b, c))
print(Rotation(a, b, c).inverse())

from sympy.physics.quantum.spin import Rotation
from sympy import pi, symbols

alpha, beta, gamma = symbols('alpha beta gamma')
print(Rotation.D(1, 1, 0, pi, pi / 2, -pi))

print(Rotation.d(1, 1, 0, pi / 2))

rot = Rotation.D(1, 1, 0, pi, pi / 2, 0)
print(rot)
print(rot.doit())

rot = Rotation.d(1, 1, 0, pi / 2)
print(rot)
print(rot.doit())

from sympy.physics.quantum.spin import JzKet, couple
from sympy.physics.quantum.tensorproduct import TensorProduct

print(couple(TensorProduct(JzKet(1, 0), JzKet(1, 1))))
print(couple(TensorProduct(JzKet(1, 1), JzKet(1, 1), JzKet(1, 0))))
print(couple(TensorProduct(JzKet(1, 1), JzKet(1, 1), JzKet(1, 0)), ((1, 3), (1, 2))))

from sympy import symbols

j1, m1, j2, m2 = symbols('j1 m1 j2 m2')
print(couple(TensorProduct(JzKet(j1, m1), JzKet(j2, m2))))

from sympy.physics.quantum.spin import JzKetCoupled, uncouple
from sympy import S

print(uncouple(JzKetCoupled(1, 0, (S(1) / 2, S(1) / 2))))

from sympy.physics.quantum.spin import JzKet

print(uncouple(JzKet(1, 0), (S(1) / 2, S(1) / 2)))
print(uncouple(JzKetCoupled(1, 0, (S(1) / 2, S(1) / 2))))
print(uncouple(JzKetCoupled(1, 1, (1, 1, 1), ((1, 3, 1), (1, 2, 1)))))
print(uncouple(JzKet(1, 1), (1, 1, 1), ((1, 3, 1), (1, 2, 1))))

from sympy import symbols

j, m, j1, j2 = symbols('j m j1 j2')
print(uncouple(JzKetCoupled(j, m, (j1, j2))))
print(uncouple(JzKet(j, m), (j1, j2)))

# State

from sympy.physics.quantum import Ket, Bra
from sympy import symbols, I

b = Bra('psi')
print(b)
print(b.hilbert_space)
print(b.is_commutative)

print(b.dual)
print(b.dual_class())

n, m = symbols('n m')
b = Bra(n, m) - I * Bra(m, n)
print(b)
print(b.subs(n, m))

k = Ket('psi')
print(k)
print(k.hilbert_space)
print(k.is_commutative)
print(k.label)
print(k.dual)
print(k.dual_class())

k0 = Ket(0)
k1 = Ket(1)
print(2 * I * k0 - 4 * k1)

from sympy.physics.quantum import OrthogonalBra, OrthogonalKet, qapply
from sympy.abc import m, n

print((OrthogonalBra(n) * OrthogonalKet(n)).doit())
print((OrthogonalBra(n) * OrthogonalKet(n + 1)).doit())
print((OrthogonalBra(n) * OrthogonalKet(m)).doit())

from sympy.physics.quantum import TimeDepBra, TimeDepKet
from sympy import symbols, I

b = TimeDepBra('psi', 't')
print(b)
print(b.time)
print(b.label)
print(b.hilbert_space)
print(b.dual)

k = TimeDepKet('psi', 't')
print(k)
print(k.time)
print(k.label)
print(k.hilbert_space)
print(k.dual)
print(k.dual_class())

from sympy import Symbol, Piecewise, pi, N
from sympy.functions import sqrt, sin
from sympy.physics.quantum.state import Wavefunction

x = Symbol('x', real=True)
n = 1
L = 1
g = Piecewise((0, x < 0), (0, x > L), (sqrt(2 // L) * sin(n * pi * x / L), True))
f = Wavefunction(g, x)
print(f.norm)
print(f.is_normalized)
p = f.prob()
print(p(0))
print(p(L))
print(p(0.5))
print(p(0.85 * L))
print(N(p(0.85 * L)))

from sympy import symbols, pi, diff
from sympy.functions import sqrt, sin
from sympy.physics.quantum.state import Wavefunction

x, L = symbols('x L', positive=True)
n = symbols('n', integer=True, positive=True)
q = sqrt(2 / L) * sin(n * pi * x / L)
f = Wavefunction(g, (x, 0, L))
print(f.norm)
print(f(L + 1))
print(f(L - 1))
print(f(-1))
print(f(0.85))
print(f(0.85, n=1, L=1))
print(f.is_commutative)

expr = x ** 2
f = Wavefunction(expr, 'x')
print(type(f.variables[0]))
print(diff(f, x))

x, y = symbols('x, y')
f = Wavefunction(x ** 2, x)
print(f.expr)

x, L = symbols('x,L', positive=True)
n = symbols('n', integer=True, positive=True)
g = sqrt(2 / L) * sin(n * pi * x / L)
f = Wavefunction(g, (x, 0, L))
print(f.is_normalized)

f = Wavefunction(x ** 2, (x, 0, 1))
print(f.limits)

f = Wavefunction(x ** 2, x)
print(f.limits)

f = Wavefunction(x ** 2, y ** 2, x, (y, -1, 2))
print(f.limits)

g = sqrt(2 / L) * sin(n * pi * x / L)
f = Wavefunction(g, (x, 0, L))
print(f.norm)

g = sin(n * pi * x / L)
f = Wavefunction(g, (x, 0, L))
print(f.norm)
print(f.normalize())
print(f.prob())

f = Wavefunction(x * y, x, y)
print(f.variables)
g = Wavefunction(x * y, x)
print(g.variables)

# Quantum Computation

# Circuit Plot

# Gates

from sympy.physics.quantum.gate import CNOT
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.qubit import Qubit

c = CNOT(1, 0)
print(qapply(c * Qubit('10')))  # note that qubits are indexed from roght to left

from sympy.physics.quantum.gate import HadamardGate
from sympy.physics.quantum.qapply import qapply

print(qapply(HadamardGate(0) * Qubit('1')))

# Hadamard on bell state, applied on 2 qubits
psi = 1 / sqrt(2) * (Qubit('00') + Qubit('11'))
print(qapply(HadamardGate(0) * HadamardGate(1) * psi))

# Grover's Algorithm
from sympy.physics.quantum.qubit import IntQubit
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.grover import OracleGate, apply_grover, superposition_basis, grover_iteration

f = lambda qubits: qubits == IntQubit(2)
v = OracleGate(2, f)
print(qapply(v * IntQubit(2)))
print(qapply(v * IntQubit(3)))
print(qapply(apply_grover(f, 2)))

numqubits = 2
basis_states = superposition_basis(numqubits)
v = OracleGate(numqubits, f)
print(qapply(grover_iteration(basis_states, v)))
print(superposition_basis(2))

# QFT

# Qubit
q = IntQubit(5)
print(q)
q = IntQubit(Qubit('101'))
print(q)
print(q.as_int())
print(q.nqubits)
print(q.qubit_values)
print(Qubit(q))

print(IntQubit(1, 1))
print(IntQubit(1, nqubits=1))
a = 1
print(IntQubit(a, nqubits=1))
print(Qubit(0, 0, 0))
q = Qubit('0101')
print(q)
print(q.nqubits)
print(len(q))
print(q.dimension)
print(q.qubit_values)

print(q.flip(1))

from sympy.physics.quantum.dagger import Dagger

print(Dagger(q))
print(type(Dagger(q)))
ip = Dagger(q) * q
print(ip)
print(ip.doit())

from sympy.physics.quantum.qubit import matrix_to_qubit, Qubit
from sympy.physics.quantum.gate import Z
from sympy.physics.quantum.represent import represent

q = Qubit('01')
print(matrix_to_qubit(represent(q)))

from sympy.physics.quantum.qubit import Qubit, measure_all, measure_partial
from sympy.physics.quantum.gate import H, X, Y, Z
from sympy.physics.quantum.qapply import qapply

c = H(0) * H(1) * Qubit('00')
print(c)
q = qapply(c)
print(measure_all(q))

print(measure_partial(q, (0,)))

# Shor's Algorithm

# Analytical Solutions

# Particle in a Box
