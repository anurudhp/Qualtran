#  Copyright 2023 Google LLC
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
import pytest

import qualtran.testing as qlt_testing
from qualtran import Bloq, BloqBuilder, QUInt
from qualtran.bloqs.basic_gates import (
    CZ,
    IntEffect,
    IntState,
    MeasZ,
    MinusState,
    OneEffect,
    OneState,
    PlusState,
    XGate,
    ZeroEffect,
    ZeroState,
    ZGate,
)
from qualtran.bloqs.basic_gates.z_basis import (
    _int_effect,
    _int_state,
    _one_effect,
    _one_state,
    _zero_effect,
    _zero_state,
    _zgate,
)
from qualtran.cirq_interop.t_complexity_protocol import t_complexity, TComplexity
from qualtran.resource_counting import GateCounts, get_cost_value, QECGatesCost
from qualtran.resource_counting.classify_bloqs import bloq_is_clifford


def test_zero_state(bloq_autotester):
    bloq_autotester(_zero_state)


def test_zero_effect(bloq_autotester):
    bloq_autotester(_zero_effect)


def test_one_state(bloq_autotester):
    bloq_autotester(_one_state)


def test_one_effect(bloq_autotester):
    bloq_autotester(_one_effect)


def test_zgate(bloq_autotester):
    bloq_autotester(_zgate)


def test_int_state(bloq_autotester):
    bloq_autotester(_int_state)


def test_int_effect(bloq_autotester):
    bloq_autotester(_int_effect)


def test_zero_state_manual():
    bloq = ZeroState()
    assert str(bloq) == '|0>'
    assert not bloq.bit
    vector = bloq.tensor_contract()
    should_be = np.array([1, 0])
    np.testing.assert_allclose(should_be, vector)

    (x,) = bloq.call_classically()
    assert x == 0

    assert get_cost_value(bloq, QECGatesCost()) == GateCounts()


def test_multiq_zero_state():
    # Verifying the attrs trickery that I can plumb through *some*
    # of the attributes but pre-specify others.
    with pytest.raises(NotImplementedError):
        _ = ZeroState(n=10)


def test_one_state_manual():
    bloq = OneState()
    assert bloq.bit
    assert bloq.state
    vector = bloq.tensor_contract()
    should_be = np.array([0, 1])
    np.testing.assert_allclose(should_be, vector)

    (x,) = bloq.call_classically()
    assert x == 1


def test_zero_effect_manual():
    bloq = ZeroEffect()
    vector = bloq.tensor_contract()

    # Note: we don't do "column vectors" or anything for kets.
    # Everything is squeezed. Keep track manually or use compositebloq.
    should_be = np.array([1, 0])
    np.testing.assert_allclose(should_be, vector)

    ret = bloq.call_classically(q=0)
    assert ret == ()

    with pytest.raises(AssertionError):
        bloq.call_classically(q=1)

    with pytest.raises(ValueError, match=r'Bad QBit\(\) value \[0\, 0\, 0\]'):
        bloq.call_classically(q=[0, 0, 0])  # type: ignore[arg-type]


def test_one_effect_manual():
    bloq = OneEffect()
    vector = bloq.tensor_contract()

    # Note: we don't do "column vectors" or anything for kets.
    # Everything is squeezed. Keep track manually or use compositebloq.
    should_be = np.array([0, 1])
    np.testing.assert_allclose(should_be, vector)

    ret = bloq.call_classically(q=1)
    assert ret == ()

    with pytest.raises(AssertionError):
        bloq.call_classically(q=0)


@pytest.mark.parametrize('bit', [False, True])
def test_zero_state_effect(bit):
    bb = BloqBuilder()

    if bit:
        state: Bloq = OneState()
        eff: Bloq = OneEffect()
    else:
        state = ZeroState()
        eff = ZeroEffect()

    q0 = bb.add(state)
    bb.add(eff, q=q0)
    cbloq = bb.finalize()
    val = cbloq.tensor_contract()

    should_be = 1
    np.testing.assert_allclose(should_be, val)

    res = cbloq.call_classically()
    assert res == ()


def test_int_state_manual():
    k = IntState(255, bitsize=8)
    assert str(k) == '|255>'
    (val,) = k.call_classically()
    assert val == 255

    with pytest.raises(ValueError):
        _ = IntState(255, bitsize=7)
    with pytest.raises(ValueError):
        _ = IntState(-1, bitsize=8)

    np.testing.assert_allclose(k.tensor_contract(), k.decompose_bloq().tensor_contract())


def test_int_effect_manual():
    k = IntEffect(255, bitsize=8)
    assert str(k) == '<255|'
    ret = k.call_classically(val=255)
    assert ret == ()

    with pytest.raises(AssertionError):
        k.call_classically(val=245)

    qlt_testing.assert_valid_bloq_decomposition(k)
    np.testing.assert_allclose(k.tensor_contract(), k.decompose_bloq().tensor_contract())


def test_to_cirq():
    bb = BloqBuilder()
    q = bb.add(ZeroState())
    q = bb.add(ZGate(), q=q)
    cbloq = bb.finalize(q=q)
    circuit = cbloq.to_cirq_circuit()
    cirq.testing.assert_has_diagram(circuit, "_c(0): ───Z───")
    vec1 = cbloq.tensor_contract()
    vec2 = cirq.final_state_vector(circuit)
    np.testing.assert_allclose(vec1, vec2)

    bb = BloqBuilder()
    q = bb.add(OneState())
    q = bb.add(ZGate(), q=q)
    cbloq = bb.finalize(q=q)
    circuit = cbloq.to_cirq_circuit()
    cirq.testing.assert_has_diagram(circuit, "_c(0): ───X───Z───")
    vec1 = cbloq.tensor_contract()
    vec2 = cirq.final_state_vector(circuit)
    np.testing.assert_allclose(vec1, vec2)


def test_pl_interop():
    import pennylane as qml

    bloq = ZGate()
    pl_op_from_bloq = bloq.as_pl_op(wires=[0])
    pl_op = qml.Z(wires=[0])
    assert pl_op_from_bloq == pl_op

    matrix = pl_op.matrix()
    should_be = bloq.tensor_contract()
    np.testing.assert_allclose(should_be, matrix)


def test_zgate_manual():
    z = ZGate()

    np.testing.assert_allclose(cirq.unitary(cirq.Z), z.tensor_contract())
    (op,) = list(z.as_composite_bloq().to_cirq_circuit().all_operations())
    assert op.gate == cirq.Z

    assert bloq_is_clifford(z)
    assert t_complexity(z) == TComplexity(clifford=1)


def test_cz_manual():
    cz = CZ()

    np.testing.assert_allclose(cirq.unitary(cirq.CZ), cz.tensor_contract())
    (op,) = list(cz.as_composite_bloq().to_cirq_circuit().all_operations())
    assert op.gate == cirq.CZ

    assert bloq_is_clifford(cz)

    assert ZGate().controlled() == CZ()
    assert t_complexity(cz) == TComplexity(clifford=1)

    with pytest.raises(ValueError, match='.*phase.*'):
        cz.call_classically(q1=1, q2=1)


def test_cz_phased_classical():
    cz = CZ()
    from qualtran.simulation.classical_sim import do_phased_classical_simulation

    final_vals, phase = do_phased_classical_simulation(cz, {'q1': 0, 'q2': 1})
    assert final_vals['q1'] == 0
    assert final_vals['q2'] == 1
    assert phase == 1

    final_vals, phase = do_phased_classical_simulation(cz, {'q1': 1, 'q2': 1})
    assert final_vals['q1'] == 1
    assert final_vals['q2'] == 1
    assert phase == -1

    bb = BloqBuilder()
    q1 = bb.add(ZeroState())
    q2 = bb.add(ZeroState())
    q1 = bb.add(XGate(), q=q1)
    q2 = bb.add(XGate(), q=q2)
    q1, q2 = bb.add(CZ(), q1=q1, q2=q2)
    cbloq = bb.finalize(q1=q1, q2=q2)
    final_vals, phase = do_phased_classical_simulation(cbloq, {})
    assert final_vals['q1'] == 1
    assert final_vals['q2'] == 1
    assert phase == -1


def test_meas_z_supertensor():
    with pytest.raises(ValueError, match=r'.*superoperator.*'):
        MeasZ().tensor_contract()

    # Zero -> Zero
    bb = BloqBuilder()
    q = bb.add(ZeroState())
    c = bb.add(MeasZ(), q=q)
    cbloq = bb.finalize(c=c)
    rho = cbloq.tensor_contract(superoperator=True)
    should_be = np.outer([1, 0], [1, 0])
    np.testing.assert_allclose(rho, should_be, atol=1e-8)

    # One -> One
    bb = BloqBuilder()
    q = bb.add(OneState())
    c = bb.add(MeasZ(), q=q)
    cbloq = bb.finalize(c=c)
    rho = cbloq.tensor_contract(superoperator=True)
    should_be = np.outer([0, 1], [0, 1])
    np.testing.assert_allclose(rho, should_be, atol=1e-8)

    # Plus -> mixture
    bb = BloqBuilder()
    q = bb.add(PlusState())
    c = bb.add(MeasZ(), q=q)
    cbloq = bb.finalize(c=c)
    rho = cbloq.tensor_contract(superoperator=True)
    should_be = np.diag([0.5, 0.5])
    np.testing.assert_allclose(rho, should_be, atol=1e-8)

    # Minus -> mixture
    bb = BloqBuilder()
    q = bb.add(MinusState())
    c = bb.add(MeasZ(), q=q)
    cbloq = bb.finalize(c=c)
    rho = cbloq.tensor_contract(superoperator=True)
    should_be = np.diag([0.5, 0.5])
    np.testing.assert_allclose(rho, should_be, atol=1e-8)


def test_meas_z_classical():
    bb = BloqBuilder()
    q = bb.add(IntState(val=52, bitsize=8))
    qs = bb.split(q)
    for i in range(8):
        qs[i] = bb.add(MeasZ(), q=qs[i])
    cbloq = bb.finalize(outs=qs)
    (ret,) = cbloq.call_classically()
    assert list(ret) == QUInt(8).to_bits(52)  # type: ignore[arg-type]
