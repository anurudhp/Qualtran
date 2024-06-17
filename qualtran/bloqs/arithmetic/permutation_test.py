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
import numpy as np
import pytest

from qualtran import QBit
from qualtran.bloqs.arithmetic.permutation import (
    _permutation,
    _permutation_cycle,
    Permutation,
    PermutationCycle,
)
from qualtran.bloqs.basic_gates import CNOT, TGate, XGate
from qualtran.bloqs.bookkeeping import Allocate, ArbitraryClifford, Free
from qualtran.resource_counting.generalizers import ignore_split_join


def test_examples(bloq_autotester):
    if bloq_autotester.check_name == 'serialize':
        pytest.skip()

    bloq_autotester(_permutation_cycle)
    bloq_autotester(_permutation)


def test_permutation_cycle_unitary_and_call_graph():
    bloq = PermutationCycle(4, np.array([0, 1, 2]))

    np.testing.assert_allclose(
        bloq.tensor_contract(), np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    )

    _, sigma = bloq.call_graph(generalizer=ignore_split_join)
    assert sigma == {
        CNOT(): 8,
        TGate(): 16,
        ArbitraryClifford(n=2): 76,
        Allocate(QBit()): 1,
        Free(QBit()): 1,
    }


def test_permutation_unitary_and_call_graph():
    bloq = Permutation(7, np.array([1, 2, 0, 4, 3, 5, 6]))

    np.testing.assert_allclose(
        bloq.tensor_contract(),
        np.array(
            [
                [0, 0, 1, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ]
        ),
    )

    _, sigma = bloq.call_graph(generalizer=ignore_split_join)
    assert sigma == {
        CNOT(): 17,
        TGate(): 56,
        XGate(): 56,
        ArbitraryClifford(n=2): 182,
        Allocate(QBit()): 2,
        Free(QBit()): 2,
    }
