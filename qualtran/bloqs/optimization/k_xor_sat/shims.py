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
from typing import Optional

import attrs

from qualtran import Bloq, CtrlSpec, Signature
from qualtran.resource_counting import CostKey, GateCounts, QECGatesCost
from qualtran.symbolics import SymbolicInt


@attrs.frozen
class ArbitraryGate(Bloq):
    """Placeholder gate for costing

    Footnote 18, page 29:
        By “gate complexity”, we mean the total number of (arbitrary) 1- and 2-qubit gates
        used by the quantum algorithm. These gates can be further represented using a
        finite universal gate set with a logarithmic overhead.

    References:
        [Quartic quantum speedups for planted inference](https://arxiv.org/abs/2406.19378v1)
        Section 4.4.2 for algorithm. Section 2.4 for definitions and notation.
    """

    n_ctrls: SymbolicInt = 0

    @property
    def signature(self) -> 'Signature':
        return Signature.build(q=1 + self.n_ctrls)

    def my_static_costs(self, cost_key: 'CostKey'):
        if isinstance(cost_key, QECGatesCost):
            # placeholder cost: reduce controls to single bit, and use C-SU2 (say).
            return GateCounts(rotation=1, and_bloq=self.n_ctrls)

        return NotImplemented

    def adjoint(self) -> 'Bloq':
        return self

    def get_ctrl_system(self, ctrl_spec: CtrlSpec):
        ctrl_bloq = attrs.evolve(self, n_ctrls=self.n_ctrls + ctrl_spec.num_qubits)

        return ctrl_bloq, NotImplemented


def generalize_1_2_qubit_gates(b: Bloq) -> Optional[Bloq]:
    from qualtran.bloqs.basic_gates import GlobalPhase, Identity
    from qualtran.bloqs.bookkeeping import ArbitraryClifford
    from qualtran.resource_counting.classify_bloqs import (
        bloq_is_clifford,
        bloq_is_rotation,
        bloq_is_t_like,
    )

    if bloq_is_t_like(b) or bloq_is_clifford(b) or bloq_is_rotation(b):
        return ArbitraryGate()

    if isinstance(b, ArbitraryClifford):
        return ArbitraryGate()

    if isinstance(b, (GlobalPhase, Identity)):
        return None

    return b
