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
import cirq
import numpy as np

from qualtran.bloqs.arithmetic.permutation import PermutationCycle


def test_permutation_cycle_on_basis_states():
    cycle = (0, 1, 2)
    bloq = PermutationCycle(4, cycle)

    np.testing.assert_allclose(
        cirq.unitary(bloq), np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    )


def test_permutation_bloq_on_random_permutations():
    pass
