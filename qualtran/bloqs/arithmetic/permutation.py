#  Copyright 2024 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from functools import cached_property
from typing import Dict, Iterator, Sequence, Set, TYPE_CHECKING, Union

import numpy as np
from attrs import field, frozen
from numpy.typing import NDArray

from qualtran import (
    Bloq,
    bloq_example,
    BoundedQUInt,
    DecomposeTypeError,
    QAny,
    QBit,
    QDType,
    Signature,
)
from qualtran.bloqs.basic_gates import CNOT, XGate
from qualtran.bloqs.mcmt.and_bloq import And, MultiAnd
from qualtran.simulation.classical_sim import ints_to_bits
from qualtran.symbolics import bit_length, HasLength, is_symbolic, Shaped, slen, SymbolicInt

if TYPE_CHECKING:
    import sympy

    from qualtran import BloqBuilder, Soquet, SoquetT
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator
    from qualtran.simulation.classical_sim import ClassicalValT


@frozen
class EqualK(Bloq):
    r"""Maps |x>|b> to |x>|b \oplus (x == k)> for a constant k"""
    dtype: QDType
    k: SymbolicInt

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(x=self.dtype, result=QBit())

    def is_symbolic(self):
        return is_symbolic(self.k, self.dtype)

    def build_composite_bloq(
        self, bb: 'BloqBuilder', x: 'Soquet', result: 'Soquet'
    ) -> Dict[str, 'SoquetT']:
        if self.is_symbolic():
            raise DecomposeTypeError(f"cannot decompose symbolic {self}")

        num_qubits = self.dtype.num_qubits
        (bits_k,) = ints_to_bits(self.k, self.dtype.num_qubits)

        if num_qubits == 1:
            if self.k == 0:
                x = bb.add(XGate(), q=x)
            x, result = bb.add(CNOT(), ctrl=x, target=result)
            if self.k == 0:
                x = bb.add(XGate(), q=x)
        elif num_qubits == 2:
            control_bloq = And(bits_k[0], bits_k[1])

            xs = bb.split(x)
            xs, target = bb.add(control_bloq, ctrl=xs)
            target, result = bb.add(CNOT(), ctrl=target, target=result)
            xs = bb.add(control_bloq.adjoint(), ctrl=xs, target=target)
            x = bb.join(xs)
        else:
            control_bloq = MultiAnd(tuple(bits_k))

            xs = bb.split(x)
            xs, junk, target = bb.add(control_bloq, ctrl=xs)
            target, result = bb.add(CNOT(), ctrl=target, target=result)
            xs = bb.add(control_bloq.adjoint(), ctrl=xs, junk=junk, target=target)
            x = bb.join(xs)

        return {'x': x, 'result': result}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        if self.is_symbolic():
            num_qubits = self.dtype.num_qubits
            if not is_symbolic(num_qubits) and num_qubits <= 2:
                if num_qubits == 1:
                    return {(XGate(), 2), (CNOT(), 1)}
                else:
                    cv = ssa.new_symbol('cv')
                    return {(And(cv, cv), 1), (And(cv, cv).adjoint(), 1), (CNOT(), 1)}
            else:
                return {
                    (MultiAnd(HasLength(num_qubits)), 1),
                    (MultiAnd(HasLength(num_qubits)).adjoint(), 1),
                    (CNOT(), 1),
                }

        return super().build_call_graph(ssa)


@frozen
class XorK(Bloq):
    r"""Maps |x> to |x \oplus k> for a constant k"""
    bitsize: SymbolicInt
    k: SymbolicInt

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(x=QAny(self.bitsize))

    def is_symbolic(self):
        return is_symbolic(self.k, self.bitsize)

    def build_composite_bloq(self, bb: 'BloqBuilder', x: 'Soquet') -> Dict[str, 'SoquetT']:
        if self.is_symbolic():
            raise DecomposeTypeError(f"cannot decompose symbolic {self}")

        xs = bb.split(x)
        (bits_k,) = ints_to_bits(self.k, self.bitsize)

        for i, bit in enumerate(bits_k):
            if bit == 1:
                xs[i] = bb.add(XGate(), q=xs[i])

        return {'x': bb.join(xs)}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        if self.is_symbolic():
            return {(XGate(), self.bitsize)}

        (bits_k,) = ints_to_bits(self.k, self.bitsize)
        return {(XGate(), sum(bits_k))}


@frozen
class PermutationCycle(Bloq):
    """Apply a single permutation cycle on the basis states"""

    N: SymbolicInt
    cycle: Union[NDArray[np.integer], Shaped] = field(
        eq=lambda x: x if isinstance(x, Shaped) else tuple(x.flatten())
    )

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(q=self.dtype)

    @cached_property
    def dtype(self):
        return BoundedQUInt(self.bitsize, self.N)

    @cached_property
    def bitsize(self):
        return bit_length(self.N - 1)

    def is_symbolic(self):
        return is_symbolic(self.N, self.cycle)

    def build_composite_bloq(self, bb: 'BloqBuilder', q: 'SoquetT') -> dict[str, 'SoquetT']:
        if self.is_symbolic():
            raise DecomposeTypeError(f"cannot decompose symbolic {self}")

        a = bb.allocate(dtype=QBit())

        for k, x_k in enumerate(self.cycle):
            q, a = bb.add_t(EqualK(self.dtype, x_k), x=q, result=a)

            delta = x_k ^ self.cycle[(k + 1) % len(self.cycle)]
            a, q = bb.add_t(XorK(self.bitsize, delta).controlled(), ctrl=a, x=q)

        q, a = bb.add_t(EqualK(self.dtype, self.cycle[0]), x=q, result=a)

        bb.free(a)

        return {'q': q}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        if self.is_symbolic():
            x = ssa.new_symbol('x')
            cycle_len = slen(self.cycle)
            return {
                (EqualK(self.dtype, x), cycle_len + 1),
                (XorK(self.bitsize, x).controlled(), cycle_len),
            }

        return super().build_call_graph(ssa)

    def on_classical_vals(
        self, q: Union['sympy.Symbol', 'ClassicalValT']
    ) -> dict[str, 'ClassicalValT']:
        if is_symbolic(self.cycle):
            raise ValueError("cannot simulate classically on symbolic permuation cycle")

        if q not in self.cycle:
            return q

        index = self.cycle.index(q)
        return {'q': self.cycle[(index + 1) % len(self.cycle)]}


def _decompose_permutation_into_cycles(permutation: Sequence[int]) -> Iterator[NDArray[np.integer]]:
    """Generate all non-trivial (more than one element) cycles of a permutation of [0, ..., N - 1]"""
    d = len(permutation)
    seen = np.full(d, False)

    for i in range(d):
        if seen[i]:
            continue

        idx, cycle = i, []
        while idx < d and not seen[idx]:
            seen[idx] = True
            idx = permutation[idx]
            cycle.append(idx)

        if len(cycle) > 1:
            yield np.array(cycle)


@frozen
class Permutation(Bloq):
    """Apply a permutation of [0, N - 1] on the basis states"""

    N: SymbolicInt
    permutation: Union[NDArray[np.integer], Shaped] = field(
        eq=lambda x: x if isinstance(x, Shaped) else tuple(x.flatten())
    )

    def __attrs_post_init__(self):
        N = self.N
        permutation_length = slen(self.permutation)
        if not is_symbolic(N, permutation_length):
            assert permutation_length <= N, f"{permutation_length=} out of bounds for limit {N=}"

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(q=BoundedQUInt(bit_length(self.N - 1), self.N))

    def is_symbolic(self):
        return is_symbolic(self.N, self.permutation)

    def build_composite_bloq(self, bb: 'BloqBuilder', q: 'Soquet') -> dict[str, 'SoquetT']:
        if self.is_symbolic():
            raise DecomposeTypeError(f"cannot decompose symbolic {self}")

        for cycle in _decompose_permutation_into_cycles(self.permutation):
            q = bb.add(PermutationCycle(self.N, cycle), q=q)

        return {'q': q}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        if self.is_symbolic():
            pass

        return super().build_call_graph(ssa)

    def on_classical_vals(
        self, q: Union['sympy.Symbol', 'ClassicalValT']
    ) -> dict[str, 'ClassicalValT']:
        if is_symbolic(self.permutation):
            raise ValueError("cannot simulate classically on symbolic permuation")
        return {'q': self.permutation[q] if q < len(self.permutation) else q}


@bloq_example
def _permutation_cycle() -> PermutationCycle:
    permutation_cycle = PermutationCycle(4, np.array([0, 1, 2]))
    return permutation_cycle


@bloq_example
def _permutation() -> Permutation:
    permutation = Permutation(4, np.array([1, 3, 0, 2]))
    return permutation
