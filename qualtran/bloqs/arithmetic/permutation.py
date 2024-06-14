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
    BoundedQUInt,
    CtrlSpec,
    DecomposeTypeError,
    GateWithRegisters,
    QAny,
    QBit,
    QDType,
    Signature,
)
from qualtran.bloqs.basic_gates import XGate
from qualtran.simulation.classical_sim import ints_to_bits
from qualtran.symbolics import bit_length, is_symbolic, Shaped, slen, SymbolicInt

if TYPE_CHECKING:
    import sympy

    from qualtran import BloqBuilder, Soquet, SoquetT
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator
    from qualtran.simulation.classical_sim import ClassicalValT


def _decompose_permutation_into_cycles(permutation: Sequence[int]) -> Iterator[tuple[int, ...]]:
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
            yield tuple(cycle)


@frozen
class EqualK(GateWithRegisters):
    r"""Maps |x>|b> to |x>|b \oplus (x == k)> for a constant k"""
    dtype: QDType
    k: SymbolicInt

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(x=self.dtype, result=QBit())

    def is_symbolic(self):
        return is_symbolic(self.k, self.dtype)

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        x, result = bb.add(
            XGate().controlled(ctrl_spec=CtrlSpec(qdtypes=self.dtype, cvs=self.k)),
            ctrl=soqs['input'],
            q=soqs['result'],
        )
        return {'x': x, 'result': result}


@frozen
class XorK(GateWithRegisters):
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
class PermutationCycle(GateWithRegisters):
    """Apply a single permutation cycle on the basis states"""

    N: SymbolicInt
    cycle: Union[tuple[int, ...], Shaped] = field(
        converter=lambda x: x if isinstance(x, Shaped) else tuple(x)
    )

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(q=self.dtype)

    @cached_property
    def dtype(self):
        return BoundedQUInt(self.bitsize, self.N)

    @cached_property
    def bitsize(self):
        return bit_length(self.N)

    def build_composite_bloq(self, bb: 'BloqBuilder', q: 'SoquetT') -> dict[str, 'SoquetT']:
        if self.is_symbolic():
            raise DecomposeTypeError(f"cannot decompose symbolic {self}")

        a = bb.allocate(1)

        for k, x_k in enumerate(self.cycle):
            q, a = bb.add_t(EqualK(self.dtype, x_k), x=q, result=a)

            delta = x_k ^ self.cycle[(k + 1) % len(self.cycle)]
            a, q = bb.add_t(XorK(self.bitsize, delta).controlled(), ctrl=a, x=q)

        q, a = bb.add_t(EqualK(self.dtype, self.cycle[0]), x=q, result=a)

        bb.free(a)

        return {'q': q}

    def is_symbolic(self):
        return is_symbolic(self.N, self.cycle)

    def on_classical_vals(
        self, q: Union['sympy.Symbol', 'ClassicalValT']
    ) -> dict[str, 'ClassicalValT']:
        if is_symbolic(self.cycle):
            raise ValueError("cannot simulate classically on symbolic permuation cycle")

        if q not in self.cycle:
            return q

        index = self.cycle.index(q)
        return {'q': self.cycle[(index + 1) % len(self.cycle)]}


@frozen
class Permutation(GateWithRegisters):
    """Apply a permutation of [0, N - 1] on the basis states"""

    N: SymbolicInt
    permutation: Union[NDArray, Shaped]

    def __attrs_post_init__(self):
        N = self.N
        permutation_length = slen(self.permutation)
        if not is_symbolic(N, permutation_length):
            assert permutation_length <= N, f"{permutation_length=} out of bounds for limit {N=}"

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(q=BoundedQUInt(bit_length(self.N), self.N))

    def build_composite_bloq(self, bb: 'BloqBuilder', q: 'Soquet') -> dict[str, 'SoquetT']:
        if self.is_symbolic():
            raise DecomposeTypeError(f"cannot decompose symbolic {self}")

        for cycle in _decompose_permutation_into_cycles(self.permutation):
            q = bb.add(PermutationCycle(self.N, cycle), q=q)

        return {'q': q}

    def is_symbolic(self):
        return is_symbolic(self.N, self.permutation)

    def on_classical_vals(
        self, q: Union['sympy.Symbol', 'ClassicalValT']
    ) -> dict[str, 'ClassicalValT']:
        if is_symbolic(self.permutation):
            raise ValueError("cannot simulate classically on symbolic permuation")
        return {'q': self.permutation[q] if q < len(self.permutation) else q}
