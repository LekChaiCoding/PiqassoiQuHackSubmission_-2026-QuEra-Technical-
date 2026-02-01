"""
7-16-testing.py - Comparison of d=3 [[7,1,3]] vs d=5 [[17,1,5]] Distillation
==============================================================================

This module compares the noise resilience of two magic state distillation circuits:
- d=3 [[7,1,3]] Steane code (7 qubits) - from noisetesting.py
- d=5 [[17,1,5]] code (17 qubits) - from d5_piqasso_16_qubits.py

Tests:
- Scaling coefficient analysis comparing both circuits
- CZ unpaired gate parameters (cz_unpaired_gate_px/py/pz)
- Mover/Sitter parameters (TwoZone model only)

Parts include code attributed to bloqade-circuit (https://github.com/QuEraComputing/bloqade-circuit/tree/main).
Apache License 2.0 with LLVM Exceptions. Modified by Team Piqasso, 1 Feb. 2026.
"""

from bloqade import squin
from bloqade.types import Qubit
from kirin.dialects import ilist
from bloqade.cirq_utils import load_circuit, emit_circuit, noise, parallelize, transpile
import bloqade.stim
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from typing import Literal, List, Tuple, Optional, Sequence, cast, Iterable
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from scipy.optimize import curve_fit
from collections import deque, defaultdict
from dataclasses import dataclass
import cirq
from cirq.circuits.qasm_output import QasmUGate

# Optimize for parallel execution
NUM_WORKERS = min(os.cpu_count() or 4, 10)


# ============================================================================
# TWO-ZONE NOISE MODEL HELPER FUNCTIONS (from a3_twozone.ipynb)
# ============================================================================

Slot = Tuple[int, int]
Swap = Tuple[Slot, Slot]


def _get_equivalent_swaps_greedy(source, target):
    """Greedy algorithm to find swaps that transform source configuration to target."""
    src = _expand(source)
    tgt = _expand(target)

    want = defaultdict(deque)
    for i, row in enumerate(tgt):
        for j, val in enumerate(row):
            want[val].append((i, j))

    pos_to_val = {(i, j): src[i][j] for i in range(len(src)) for j in range(len(src[i]))}

    have = defaultdict(deque)
    for slot, val in pos_to_val.items():
        have[val].append(slot)

    swaps = []

    for i, row in enumerate(tgt):
        for j, desired in enumerate(row):
            slot = (i, j)
            cur = pos_to_val[slot]
            if cur == desired:
                continue

            if not have[desired]:
                continue
            other = have[desired].popleft()

            swaps.append((slot, other))

            v1 = pos_to_val[slot]
            v2 = pos_to_val[other]

            pos_to_val[slot], pos_to_val[other] = v2, v1

            have[v1].append(other)
            have[desired].append(slot)

    return swaps


def _get_qargs_from_moment(moment: cirq.Moment):
    """Returns a list of qubit arguments from all operations in a Cirq moment."""
    return [op.qubits for op in moment.operations]


def _flatten_qargs(list_qubs: Sequence[Tuple[cirq.Qid, ...]]) -> List[cirq.Qid]:
    """Flattens a list of qargs tuples."""
    return [item for tup in list_qubs for item in tup]


def _qargs_to_qidxs(qargs: List[Tuple[cirq.LineQubit, ...]]) -> List[Tuple[int, ...]]:
    """Transforms list of qargs into a list of tuples of integers."""
    return [tuple(x.x for x in tup) for tup in qargs]


def _numpy_complement(subset: np.ndarray, full: np.ndarray) -> np.ndarray:
    """Returns elements in full that are not in subset."""
    mask = ~np.isin(full, subset)
    return full[mask]


def _intersect_by_structure(
    reference: List[Tuple[int, ...]], target: List[Tuple[int, ...]]
) -> List[Tuple[int, ...]]:
    target_set = set(val for t in target for val in t)
    result = []
    for tup in reference:
        filtered = tuple(val for val in tup if val in target_set)
        result.append(filtered)
    return result


def _expand(
    data: Sequence[Tuple[Optional[int], ...]], capacity: int = 2
) -> List[List[Optional[int]]]:
    """Pad each tuple to have exactly capacity slots, using None."""
    return [list(t) + [None] * (capacity - len(t)) for t in data]


def _greedy_unique_packing(data: List[int]) -> List[List[int]]:
    remaining = deque(data)
    result = []

    while remaining:
        used = set()
        group = []
        i = 0
        length = len(remaining)

        while i < length:
            item = remaining.popleft()
            if item not in used:
                group.append(item)
                used.add(item)
            else:
                remaining.append(item)
            i += 1

        result.append(group)

    return result


def _get_swap_move_qidxs(
    swaps: List[Swap], init_qidxs: Sequence[Tuple[Optional[int], ...]]
) -> List[List[int]]:
    swap_init_qidxs = _expand(init_qidxs)
    moved_qidxs = []

    for (i1, j1), (i2, j2) in swaps:
        first_idx = swap_init_qidxs[i1][j1]
        sec_idx = swap_init_qidxs[i2][j2]

        if first_idx is not None:
            moved_qidxs.append(first_idx)
        if sec_idx is not None:
            moved_qidxs.append(sec_idx)

        swap_init_qidxs[i1][j1], swap_init_qidxs[i2][j2] = sec_idx, first_idx

    return _greedy_unique_packing(moved_qidxs)


def _pad_with_empty_tups(
    target: List[Tuple[int, ...]], nqubs: int
) -> List[Tuple[int, ...]]:
    """Pad target list with empty tuples to reach nqubs length. Returns a copy."""
    result = list(target)
    while len(result) < nqubs:
        result.append(())
    return result


def _add_noise_to_swaps(
    swaps: List[Swap],
    init_qidxs: List[Tuple[int, ...]],
    move_noise: cirq.AsymmetricDepolarizingChannel,
    sitter_noise: cirq.AsymmetricDepolarizingChannel,
    nqubs: int,
):
    """Applies move noise to qubits that need to be swapped."""
    built_circuit = cirq.Circuit()
    nqubs_idxs = np.arange(nqubs)

    batches_move_qidxs = _get_swap_move_qidxs(swaps, init_qidxs)

    for batch in batches_move_qidxs:
        built_moment = cirq.Moment()
        non_mov_qidxs = _numpy_complement(np.array(batch), nqubs_idxs)

        for i in range(len(batch)):
            built_moment += move_noise(cirq.LineQubit(batch[i]))
        for j in range(len(non_mov_qidxs)):
            built_moment += sitter_noise(cirq.LineQubit(non_mov_qidxs[j]))

        built_circuit.append(built_moment)

    return built_circuit


def _extract_u3_and_cz_qargs(moment: cirq.Moment):
    """Extracts qargs for u3 and CZ gates from a Cirq moment."""
    result = {"u3": [], "cz": [], "angles": []}

    for op in moment.operations:
        if isinstance(op.gate, QasmUGate):
            result["u3"].append(op.qubits)
            gate = cast(QasmUGate, op.gate)
            angles = (gate.theta, gate.phi, gate.lmda)
            result["angles"].append(angles)
        elif isinstance(op.gate, cirq.PhasedXZGate):
            result["u3"].append(op.qubits)
            gate = cast(cirq.PhasedXZGate, op.gate)
            angles = (gate.x_exponent, gate.z_exponent, gate.axis_phase_exponent)
            result["angles"].append(angles)
        elif isinstance(op.gate, cirq.CZPowGate):
            result["cz"].append(op.qubits)

    return result


def _get_gate_error_channel(
    moment: cirq.Moment,
    sq_loc_rates: np.ndarray,
    sq_glob_rates: np.ndarray,
    two_qubit_pauli: cirq.Gate,
    unp_cz_rates: np.ndarray,
    nqubs: int,
):
    """Applies gate errors to the circuit."""
    gates_in_layer = _extract_u3_and_cz_qargs(moment)
    new_moments = cirq.Circuit()

    if gates_in_layer["cz"] == []:
        angles = gates_in_layer["angles"]
        is_global = (
            len(angles) > 0 and
            all(np.all(np.isclose(element, angles[0])) for element in angles) and
            nqubs == len(gates_in_layer["u3"])
        )
        if is_global:
            pauli_channel = cirq.AsymmetricDepolarizingChannel(
                p_x=sq_glob_rates[0], p_y=sq_glob_rates[1], p_z=sq_glob_rates[2]
            )
            for qub in gates_in_layer["u3"]:
                new_moments.append(pauli_channel(qub[0]))
        else:
            pauli_channel = cirq.AsymmetricDepolarizingChannel(
                p_x=sq_loc_rates[0], p_y=sq_loc_rates[1], p_z=sq_loc_rates[2]
            )
            for qub in gates_in_layer["u3"]:
                new_moments.append(pauli_channel(qub[0]))
    else:
        loc_rot_pauli_channel = cirq.AsymmetricDepolarizingChannel(
            p_x=sq_loc_rates[0], p_y=sq_loc_rates[1], p_z=sq_loc_rates[2]
        )
        unp_cz_pauli_channel = cirq.AsymmetricDepolarizingChannel(
            p_x=unp_cz_rates[0], p_y=unp_cz_rates[1], p_z=unp_cz_rates[2]
        )

        for qub in gates_in_layer["cz"]:
            new_moments.append(two_qubit_pauli.on(qub[0], qub[1]))

        for qub in gates_in_layer["u3"]:
            new_moments.append(unp_cz_pauli_channel(qub[0]))
            new_moments.append(loc_rot_pauli_channel(qub[0]))

    return new_moments


def _add_move_and_sitter_channels(
    ref_qargs: Sequence[Tuple[cirq.Qid, ...]] | None,
    tar_qargs: Sequence[Tuple[cirq.Qid, ...]],
    built_moment: cirq.Moment,
    qub_reg: Sequence[cirq.Qid],
    sitter_pauli_channel: cirq.Gate,
    move_pauli_channel: cirq.Gate,
):
    """Adds move and sitter noise channels."""
    flat_tar_qargs = _flatten_qargs(tar_qargs)

    if ref_qargs is None:
        flat_ref_qargs = _flatten_qargs(tar_qargs)
        bool_list = [k not in flat_tar_qargs for k in flat_ref_qargs]
    else:
        flat_ref_qargs = _flatten_qargs(ref_qargs)
        bool_list = [k in flat_tar_qargs for k in flat_ref_qargs]

    rem_qubs = []

    for i in range(len(flat_ref_qargs)):
        if not bool_list[i]:
            rem_qubs.append(flat_ref_qargs[i])
            built_moment += move_pauli_channel(flat_ref_qargs[i])

    if len(rem_qubs) >= 1:
        for k in range(len(qub_reg)):
            if qub_reg[k] not in rem_qubs:
                built_moment += sitter_pauli_channel(qub_reg[k])
        return built_moment, True
    else:
        return built_moment, False


def _get_move_error_channel_two_zoned(
    curr_moment: cirq.Moment,
    prev_moment: cirq.Moment | None,
    move_rates: np.ndarray,
    sitter_rates: np.ndarray,
    nqubs: int,
):
    """Applies move noise channels to a cirq moment."""
    curr_qargs = _get_qargs_from_moment(curr_moment)

    move_pauli_channel = cirq.AsymmetricDepolarizingChannel(
        p_x=move_rates[0], p_y=move_rates[1], p_z=move_rates[2]
    )
    sitter_pauli_channel = cirq.AsymmetricDepolarizingChannel(
        p_x=sitter_rates[0], p_y=sitter_rates[1], p_z=sitter_rates[2]
    )
    qub_reg = [cirq.LineQubit(i) for i in range(nqubs)]

    if prev_moment is None:
        new_moment = cirq.Moment()
        dumb_circ = cirq.Circuit()
        new_moment, _ = _add_move_and_sitter_channels(
            prev_moment,
            curr_qargs,
            new_moment,
            qub_reg,
            sitter_pauli_channel,
            move_pauli_channel,
        )
        dumb_circ.append(new_moment)
    else:
        prev_qargs = _get_qargs_from_moment(prev_moment)
        new_moment = cirq.Moment()
        dumb_circ = cirq.Circuit()
        new_moment, _ = _add_move_and_sitter_channels(
            prev_qargs,
            curr_qargs,
            new_moment,
            qub_reg,
            sitter_pauli_channel,
            move_pauli_channel,
        )
        dumb_circ.append(new_moment)

        new_moment = cirq.Moment()
        new_moment, _ = _add_move_and_sitter_channels(
            curr_qargs,
            prev_qargs,
            new_moment,
            qub_reg,
            sitter_pauli_channel,
            move_pauli_channel,
        )
        dumb_circ.append(new_moment)

        prev_qidxs = _qargs_to_qidxs(prev_qargs)
        curr_qidxs = _qargs_to_qidxs(curr_qargs)

        intsc_rev = _intersect_by_structure(prev_qidxs, curr_qidxs)
        intsc_fow = _intersect_by_structure(curr_qidxs, prev_qidxs)

        swaps = _get_equivalent_swaps_greedy(
            _pad_with_empty_tups(intsc_rev, nqubs),
            _pad_with_empty_tups(intsc_fow, nqubs),
        )

        swap_noise_circ = _add_noise_to_swaps(
            swaps, _pad_with_empty_tups(intsc_rev, nqubs), move_pauli_channel, sitter_pauli_channel, nqubs
        )
        dumb_circ.append(swap_noise_circ)

    return dumb_circ


# ============================================================================
# CUSTOM TWO-ZONE NOISE MODEL
# ============================================================================

@dataclass(frozen=True)
class CustomGeminiTwoZoneNoiseModel(noise.model.GeminiNoiseModelABC):
    """Custom implementation of GeminiTwoZoneNoiseModel with proper move/sitter noise."""
    
    def noisy_moments(
        self, moments: Iterable[cirq.Moment], system_qubits: Sequence[cirq.Qid]
    ) -> Sequence[cirq.OP_TREE]:
        """Adds stateful noise to a series of moments."""
        if self.check_input_circuit:
            self.validate_moments(moments)

        moments = list(moments)

        if len(moments) == 0:
            return []

        nqubs = len(system_qubits)
        noisy_moment_list = []

        prev_moment: cirq.Moment | None = None

        for i in range(len(moments)):
            noisy_moment_list.extend(
                [
                    moment
                    for moment in _get_move_error_channel_two_zoned(
                        moments[i],
                        prev_moment,
                        np.array(self.mover_pauli_rates),
                        np.array(self.sitter_pauli_rates),
                        nqubs,
                    ).moments
                    if len(moment) > 0
                ]
            )

            noisy_moment_list.append(moments[i])

            noisy_moment_list.extend(
                [
                    moment
                    for moment in _get_gate_error_channel(
                        moments[i],
                        np.array(self.local_pauli_rates),
                        np.array(self.global_pauli_rates),
                        self.two_qubit_pauli,
                        np.array(self.cz_unpaired_pauli_rates),
                        nqubs,
                    ).moments
                    if len(moment) > 0
                ]
            )

            prev_moment = moments[i]

        return noisy_moment_list


def _transform_circuit_two_zone(
    circuit: cirq.Circuit,
    model: cirq.NoiseModel,
    to_native_gateset: bool = True,
) -> cirq.Circuit:
    """Transform an input circuit with two-zone noise model."""
    system_qubits = sorted(circuit.all_qubits())
    
    if to_native_gateset:
        native_circuit = transpile(circuit)
    else:
        native_circuit = circuit

    noisy_circuit = cirq.Circuit()
    for op_tree in model.noisy_moments(native_circuit, system_qubits):
        noisy_circuit += cirq.Circuit(op_tree)

    return noisy_circuit


# ============================================================================
# MODEL SELECTION GLOBALS
# ============================================================================

_USE_TWO_ZONE = False


def get_noise_model_class():
    """Get the appropriate noise model class."""
    if _USE_TWO_ZONE:
        return CustomGeminiTwoZoneNoiseModel
    return noise.GeminiOneZoneNoiseModel


def _get_noise_model_class(use_two_zone):
    """Worker-safe: Get the appropriate noise model class."""
    if use_two_zone:
        return CustomGeminiTwoZoneNoiseModel
    return noise.GeminiOneZoneNoiseModel


def get_model_name():
    """Get the display name of the current model."""
    return "CustomGeminiTwoZoneNoiseModel" if _USE_TWO_ZONE else "GeminiOneZoneNoiseModel"


# ============================================================================
# D=3 [[7,1,3]] CIRCUIT DEFINITIONS (from noisetesting.py)
# ============================================================================

@squin.kernel
def magicstateprep_d3(qubits, ind):
    """Magic state preparation for d=3."""
    squin.t(qubits[ind])


@squin.kernel
def injection_d3(q: ilist.IList[Qubit, Literal[7]]):
    """Apply magic-state injection to the 7-qubit register (d=3 Steane code)."""
    squin.reset(q[0:2])
    squin.reset(q[3:7])
    magicstateprep_d3(q, 2)
    
    for j in (3, 1, 0, 6, 4, 5):
        squin.ry(-pi / 2, q[j])

    squin.cz(q[1], q[0])
    squin.cz(q[6], q[4])
    squin.cz(q[5], q[2])

    squin.ry(pi / 2, q[2])

    squin.cz(q[3], q[6])
    squin.cz(q[0], q[5])
    squin.cz(q[4], q[2])

    for j in (0, 6, 4, 5, 2):
        squin.ry(pi / 2, q[j])

    squin.cz(q[3], q[1])
    squin.cz(q[0], q[6])
    squin.cz(q[4], q[5])

    squin.ry(pi / 2, q[1])
    squin.ry(pi / 2, q[0])
    squin.ry(pi / 2, q[4])
    squin.z(q[3])
    squin.x(q[0])
    squin.x(q[1])
    squin.x(q[3])


@squin.kernel
def steane_encode_zero_on(q: ilist.IList[Qubit, Literal[7]]):
    """Encode |0⟩^7 → |0⟩_L on the 7-qubit register q (Steane [[7,1,3]])."""
    squin.h(q[0])
    squin.h(q[1])
    squin.h(q[3])
    squin.cx(q[0], q[4])
    squin.cx(q[1], q[2])
    squin.cx(q[3], q[5])
    squin.cx(q[0], q[6])
    squin.cx(q[3], q[4])
    squin.cx(q[1], q[5])
    squin.cx(q[0], q[2])
    squin.cx(q[5], q[6])


@squin.kernel
def steane_encode_plus_on(q: ilist.IList[Qubit, Literal[7]]):
    """Encode 7 qubits as |+⟩_L: first |0⟩_L then transversal H."""
    steane_encode_zero_on(q)
    for i in range(7):
        squin.h(q[i])


@squin.kernel
def d3_circuit():
    """d=3 [[7,1,3]] A3 circuit - Fault-tolerant Steane syndrome extraction."""
    q = squin.qalloc(21)
    
    injection_d3(q[0:7])
    
    steane_encode_plus_on(q[7:14])
    for i in range(7):
        squin.cx(q[i], q[i+7])
    
    steane_encode_zero_on(q[14:21])
    for i in range(7):
        squin.cx(q[i+14], q[i])

    for i in range(7):
        squin.h(q[i+14])
    
    for i in range(7, 21):
        squin.measure(q[i])


# ============================================================================
# D=5 [[17,1,5]] CIRCUIT DEFINITIONS (from d5_piqasso_16_qubits.py)
# ============================================================================

@squin.kernel
def prepare_magic_d5(q):
    """Prepare magic state |T⟩ = H|0⟩ then T."""
    squin.h(q)
    squin.t(q)


@squin.kernel
def d5_injection(q: ilist.IList[Qubit, Literal[17]]):
    """d=5 [[17,1,5]] encoding with magic state injection."""
    # Magic state on center qubit
    prepare_magic_d5(q[7])
    
    # √Y on all ancillas (not q[7])
    squin.ry(pi/2, q[0])
    squin.ry(pi/2, q[1])
    squin.ry(pi/2, q[2])
    squin.ry(pi/2, q[3])
    squin.ry(pi/2, q[4])
    squin.ry(pi/2, q[5])
    squin.ry(pi/2, q[6])
    # skip q[7] - has magic state
    squin.ry(pi/2, q[8])
    squin.ry(pi/2, q[9])
    squin.ry(pi/2, q[10])
    squin.ry(pi/2, q[11])
    squin.ry(pi/2, q[12])
    squin.ry(pi/2, q[13])
    squin.ry(pi/2, q[14])
    squin.ry(pi/2, q[15])
    squin.ry(pi/2, q[16])
    
    # CZ Layer 1
    squin.cz(q[1], q[3])
    squin.cz(q[3], q[10])
    squin.cz(q[7], q[10])
    squin.cz(q[12], q[14])
    squin.cz(q[13], q[16])
    
    # √Y† on center
    squin.ry(-pi/2, q[7])
    squin.ry(-pi/2, q[16])
    
    # CZ Layer 2
    squin.cz(q[4], q[7])
    squin.cz(q[8], q[10])
    squin.cz(q[11], q[14])
    squin.cz(q[15], q[16])
    
    # √Y† layer
    squin.ry(-pi/2, q[4])
    squin.ry(-pi/2, q[10])
    squin.ry(-pi/2, q[14])
    squin.ry(-pi/2, q[16])
    
    # CZ Layer 3
    squin.cz(q[2], q[4])
    squin.cz(q[6], q[8])
    squin.cz(q[7], q[9])
    squin.cz(q[10], q[13])
    squin.cz(q[14], q[16])
    
    # Final √Y
    squin.ry(pi/2, q[1])
    squin.ry(pi/2, q[2])
    squin.ry(pi/2, q[3])
    squin.ry(pi/2, q[4])
    squin.ry(pi/2, q[6])
    squin.ry(pi/2, q[7])
    squin.ry(pi/2, q[8])
    squin.ry(pi/2, q[9])
    squin.ry(pi/2, q[11])
    squin.ry(pi/2, q[12])
    squin.ry(pi/2, q[14])

    # Final CZ 
    squin.cz(q[0], q[1])
    squin.cz(q[2], q[3])
    squin.cz(q[4], q[5])
    squin.cz(q[6], q[7])
    squin.cz(q[8], q[9])
    squin.cz(q[12], q[15])

    # Final √Y†
    squin.ry(-pi/2, q[0])
    squin.ry(-pi/2, q[2])
    squin.ry(-pi/2, q[5])
    squin.ry(-pi/2, q[6])
    squin.ry(-pi/2, q[8])
    squin.ry(-pi/2, q[10])
    squin.ry(-pi/2, q[12])


@squin.kernel
def d5_circuit():
    """
    Full d=5 circuit with injection and measurement.
    Uses 17 qubits for the [[17,1,5]] code.
    """
    q = squin.qalloc(17)
    
    # Apply d=5 injection
    d5_injection(q[0:17])
    
    # Measure all qubits
    for i in range(17):
        squin.measure(q[i])


# ============================================================================
# FIDELITY CALCULATION
# ============================================================================

# Syndrome indices for Steane [[7,1,3]] code (d=3)
SYND_INDICES_D3 = [[1, 3, 5, 7], [4, 5, 6, 7], [2, 3, 6, 7]]


def _check_syndrome(bits, indices):
    """Check if syndrome is triggered (product of parities == 1)."""
    parity = np.prod(1 - 2 * bits[np.array(indices) - 1])
    return parity == 1


def find_good_rate_d3(samples):
    """Calculate fidelity for d=3 circuit: fraction with all 6 syndromes triggered."""
    samples = np.asarray(samples)
    good = 0
    for sample in samples:
        x_bits, z_bits = sample[:7], sample[7:14]
        count = sum(_check_syndrome(x_bits, idx) for idx in SYND_INDICES_D3)
        count += sum(_check_syndrome(z_bits, idx) for idx in SYND_INDICES_D3)
        good += (count == 6)
    return good / len(samples)


def find_good_rate_d5(samples):
    """
    Calculate fidelity for d=5 circuit.
    
    For the d=5 [[17,1,5]] code injection, we check for consistency
    in the measured bit patterns based on CZ gate correlations.
    """
    samples = np.asarray(samples)
    n_samples = len(samples)
    
    good = 0
    for sample in samples:
        # Check parity relationships from the CZ structure
        p1 = (sample[1] ^ sample[3] ^ sample[10]) % 2
        p2 = (sample[7] ^ sample[10]) % 2
        p3 = (sample[12] ^ sample[14]) % 2
        p4 = (sample[4] ^ sample[7]) % 2
        p5 = (sample[6] ^ sample[8] ^ sample[9]) % 2
        
        parity_sum = p1 + p2 + p3 + p4 + p5
        if parity_sum <= 2:
            good += 1
    
    return good / n_samples


# ============================================================================
# SIMULATION ENGINE
# ============================================================================

# Circuit caches for both variants
_circuit_cache = {
    'd3': None,
    'd5': None
}


def _get_circuit(variant):
    """Get cached circuit for specified variant ('d3' or 'd5')."""
    global _circuit_cache
    if _circuit_cache[variant] is None:
        if variant == 'd3':
            _circuit_cache[variant] = emit_circuit(d3_circuit)
        else:
            _circuit_cache[variant] = emit_circuit(d5_circuit)
    return _circuit_cache[variant]


def run_simulation_with_noise(variant, noise_model, shots=1000):
    """Run circuit variant with noise model and return fidelity."""
    circuit = _get_circuit(variant)
    if _USE_TWO_ZONE:
        cirq_circuit = _transform_circuit_two_zone(circuit, model=noise_model)
    else:
        cirq_circuit = noise.transform_circuit(circuit, model=noise_model)
    stim_circuit = bloqade.stim.Circuit(load_circuit(cirq_circuit))
    samples = np.array(stim_circuit.compile_sampler().sample(shots=shots))
    
    if variant == 'd3':
        return find_good_rate_d3(samples)
    else:
        return find_good_rate_d5(samples)


def _run_simulation(variant, noise_model, shots, use_two_zone):
    """Worker-safe: Run circuit with noise model and return fidelity."""
    circuit = _get_circuit(variant)
    if use_two_zone:
        cirq_circuit = _transform_circuit_two_zone(circuit, model=noise_model)
    else:
        cirq_circuit = noise.transform_circuit(circuit, model=noise_model)
    stim_circuit = bloqade.stim.Circuit(load_circuit(cirq_circuit))
    samples = np.array(stim_circuit.compile_sampler().sample(shots=shots))
    
    if variant == 'd3':
        return find_good_rate_d3(samples)
    else:
        return find_good_rate_d5(samples)


# ============================================================================
# NOISE PARAMETERS (LIMITED SET)
# ============================================================================

# Base parameters: only cz_unpaired
_BASE_NOISE_PARAMS = [
    'cz_unpaired_gate_px',
    'cz_unpaired_gate_py',
    'cz_unpaired_gate_pz',
]

# Two-zone extra: mover and sitter
_TWO_ZONE_EXTRA_PARAMS = [
    'sitter_px', 'sitter_py', 'sitter_pz',
    'mover_px', 'mover_py', 'mover_pz',
]


def get_noise_params():
    """Get the appropriate noise parameters based on selected model."""
    if _USE_TWO_ZONE:
        return _BASE_NOISE_PARAMS + _TWO_ZONE_EXTRA_PARAMS
    return _BASE_NOISE_PARAMS


def _get_noise_params(use_two_zone):
    """Worker-safe: Get the appropriate noise parameters."""
    if use_two_zone:
        return _BASE_NOISE_PARAMS + _TWO_ZONE_EXTRA_PARAMS
    return _BASE_NOISE_PARAMS


def get_zero_params():
    """Get zero params dict for the selected model."""
    return {p: 0.0 for p in get_noise_params()}


def _get_zero_params(use_two_zone):
    """Worker-safe: Get zero params dict."""
    return {p: 0.0 for p in _get_noise_params(use_two_zone)}


# ============================================================================
# PARALLEL TEST WORKERS
# ============================================================================

def _run_scaling_test(args):
    """Worker function for parallel scaling coefficient tests."""
    variant, coeff, shots, use_two_zone = args
    NoiseModelClass = _get_noise_model_class(use_two_zone)
    noise_model = NoiseModelClass(scaling_factor=coeff)
    fidelity = _run_simulation(variant, noise_model, shots=shots, use_two_zone=use_two_zone)
    return (variant, coeff), fidelity


def _run_param_test(args):
    """Worker function for parallel parameter tests."""
    variant, param_name, param_val, shots, use_two_zone = args
    params = _get_zero_params(use_two_zone)
    params[param_name] = param_val
    NoiseModelClass = _get_noise_model_class(use_two_zone)
    noise_model = NoiseModelClass(**params)
    fidelity = _run_simulation(variant, noise_model, shots=shots, use_two_zone=use_two_zone)
    return (variant, param_name, param_val), fidelity


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_scaling_comparison(coeff_range=None, shots=500, verbose=True):
    """Compare scaling coefficient behavior between d3 and d5 circuits."""
    if coeff_range is None:
        coeff_range = np.linspace(0.1, 3.0, 25)
    
    # Build args for both variants
    use_two_zone = _USE_TWO_ZONE
    all_args = []
    for variant in ['d3', 'd5']:
        for coeff in coeff_range:
            all_args.append((variant, coeff, shots, use_two_zone))
    
    total = len(all_args)
    if verbose:
        print(f"  Running {total} scaling tests ({NUM_WORKERS} workers)...")
    
    results = {}
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(_run_scaling_test, args): args for args in all_args}
        for i, future in enumerate(as_completed(futures), 1):
            key, fidelity = future.result()
            results[key] = fidelity
            if verbose and i % 10 == 0:
                print(f"    Progress: {i}/{total} ({100*i/total:.0f}%)")
    
    if verbose:
        print(f"    Progress: {total}/{total} (100%)")
    
    # Organize by variant
    d3_fids = np.array([results[('d3', c)] for c in coeff_range])
    d5_fids = np.array([results[('d5', c)] for c in coeff_range])
    
    return {
        'd3': d3_fids,
        'd5': d5_fids,
        'coeffs': np.array(coeff_range)
    }


def analyze_params_comparison(iterations=15, shots=500, verbose=True):
    """Compare parameter sweeps between d3 and d5 circuits."""
    use_two_zone = _USE_TWO_ZONE
    noise_params = get_noise_params()
    
    all_args = []
    param_ranges = {}
    
    for param_name in noise_params:
        param_values = np.linspace(0, 5e-2, iterations)
        param_ranges[param_name] = param_values
        for variant in ['d3', 'd5']:
            for pv in param_values:
                all_args.append((variant, param_name, pv, shots, use_two_zone))
    
    total = len(all_args)
    if verbose:
        print(f"  Running {total} parameter tests ({NUM_WORKERS} workers)...")
    
    results = {}
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(_run_param_test, args): args for args in all_args}
        for i, future in enumerate(as_completed(futures), 1):
            key, fidelity = future.result()
            results[key] = fidelity
            if verbose and i % 20 == 0:
                print(f"    Progress: {i}/{total} ({100*i/total:.0f}%)")
    
    if verbose:
        print(f"    Progress: {total}/{total} (100%)")
    
    # Organize results
    param_results = {}
    for param_name in noise_params:
        pvals = param_ranges[param_name]
        d3_fids = np.array([results[('d3', param_name, pv)] for pv in pvals])
        d5_fids = np.array([results[('d5', param_name, pv)] for pv in pvals])
        param_results[param_name] = {
            'values': pvals,
            'd3': d3_fids,
            'd5': d5_fids
        }
    
    return param_results


# ============================================================================
# VISUALIZATION
# ============================================================================

def _exp_decay(x, A, b):
    return A * np.exp(-b * x)


def _r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0


def plot_comparison_results(scaling_results, param_results):
    """Create comparison visualization plots for d=3 vs d=5 noise analysis."""
    
    coeffs = scaling_results['coeffs']
    d3_scaling = scaling_results['d3']
    d5_scaling = scaling_results['d5']
    
    n_params = len(param_results)
    # Use 3 columns for TwoZone (9 params), but ensure proper layout
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols
    
    # Calculate figure size: wider for more columns, taller for more rows
    fig_width = 6 * n_cols
    fig_height = 5 + 4.5 * n_rows  # Extra space for top plot + param plots
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Use GridSpec for better control over subplot spacing
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(n_rows + 1, n_cols, figure=fig, 
                  height_ratios=[1.2] + [1] * n_rows,
                  hspace=0.35, wspace=0.3)
    
    # ---- Top Plot: Scaling Comparison (spans all columns) ----
    ax_top = fig.add_subplot(gs[0, :])
    
    # d=3 data (7 qubits)
    ax_top.scatter(coeffs, d3_scaling, s=80, alpha=0.8, color='#e74c3c',
                   edgecolors='black', linewidth=1, zorder=5, marker='o',
                   label='d=3 [[7,1,3]] (7 qubits)')
    
    # d=5 data (17 qubits)
    ax_top.scatter(coeffs, d5_scaling, s=80, alpha=0.8, color='#2980b9',
                   edgecolors='black', linewidth=1, zorder=5, marker='s',
                   label='d=5 [[17,1,5]] (17 qubits)')
    
    # Fit exponential decay for both
    try:
        popt_d3, _ = curve_fit(_exp_decay, coeffs, d3_scaling, 
                               p0=[d3_scaling[0], 0.5], maxfev=10000)
        fit_x = np.linspace(coeffs.min(), coeffs.max(), 100)
        r2_d3 = _r_squared(d3_scaling, _exp_decay(coeffs, *popt_d3))
        ax_top.plot(fit_x, _exp_decay(fit_x, *popt_d3), '-', color='#e74c3c',
                   linewidth=2, alpha=0.6,
                   label=f'd3: {popt_d3[0]:.2f}·exp(-{popt_d3[1]:.2f}x) R²={r2_d3:.3f}')
        print(f"  d=3 Scaling fit: A={popt_d3[0]:.4f}, b={popt_d3[1]:.4f}, R²={r2_d3:.4f}")
    except (RuntimeError, ValueError):
        print("  Warning: Could not fit d=3 scaling data")
    
    try:
        popt_d5, _ = curve_fit(_exp_decay, coeffs, d5_scaling, 
                               p0=[d5_scaling[0], 0.5], maxfev=10000)
        fit_x = np.linspace(coeffs.min(), coeffs.max(), 100)
        r2_d5 = _r_squared(d5_scaling, _exp_decay(coeffs, *popt_d5))
        ax_top.plot(fit_x, _exp_decay(fit_x, *popt_d5), '-', color='#2980b9',
                   linewidth=2, alpha=0.6,
                   label=f'd5: {popt_d5[0]:.2f}·exp(-{popt_d5[1]:.2f}x) R²={r2_d5:.3f}')
        print(f"  d=5 Scaling fit: A={popt_d5[0]:.4f}, b={popt_d5[1]:.4f}, R²={r2_d5:.4f}")
    except (RuntimeError, ValueError):
        print("  Warning: Could not fit d=5 scaling data")
    
    ax_top.set_xlabel('Scaling Coefficient', fontsize=11, fontweight='bold')
    ax_top.set_ylabel('Fidelity', fontsize=11, fontweight='bold')
    ax_top.set_title(f'd=3 vs d=5 Scaling Comparison ({get_model_name()})', 
                    fontsize=12, fontweight='bold', pad=10)
    ax_top.set_ylim([0, 1.05])
    ax_top.grid(True, linestyle='--', alpha=0.4)
    ax_top.legend(fontsize=9, loc='upper right', framealpha=0.9)
    
    # ---- Parameter Sweep Comparison Plots ----
    # Cleaner display names
    display_names = {
        'cz_unpaired_gate_px': 'CZ Unpaired Px',
        'cz_unpaired_gate_py': 'CZ Unpaired Py',
        'cz_unpaired_gate_pz': 'CZ Unpaired Pz',
        'sitter_px': 'Sitter Px',
        'sitter_py': 'Sitter Py',
        'sitter_pz': 'Sitter Pz',
        'mover_px': 'Mover Px',
        'mover_py': 'Mover Py',
        'mover_pz': 'Mover Pz',
    }
    
    param_items = list(param_results.items())
    
    for i, (param_name, data) in enumerate(param_items):
        row = 1 + i // n_cols
        col = i % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        param_values = data['values']
        d3_fids = data['d3']
        d5_fids = data['d5']
        
        # d=3 scatter
        ax.scatter(param_values, d3_fids, s=40, alpha=0.7, color='#e74c3c',
                   edgecolors='black', linewidth=0.5, zorder=3, marker='o',
                   label='d=3')
        
        # d=5 scatter
        ax.scatter(param_values, d5_fids, s=40, alpha=0.7, color='#2980b9',
                   edgecolors='black', linewidth=0.5, zorder=3, marker='s',
                   label='d=5')
        
        # Fit exponential for both
        if np.std(d3_fids) > 0.01:
            try:
                popt, _ = curve_fit(_exp_decay, param_values, d3_fids,
                                   p0=[d3_fids[0], 50], maxfev=10000)
                fit_x = np.linspace(param_values.min(), param_values.max(), 100)
                ax.plot(fit_x, _exp_decay(fit_x, *popt), '-', color='#e74c3c',
                       linewidth=1.5, alpha=0.5, zorder=2)
            except (RuntimeError, ValueError):
                pass
        
        if np.std(d5_fids) > 0.01:
            try:
                popt, _ = curve_fit(_exp_decay, param_values, d5_fids,
                                   p0=[d5_fids[0], 50], maxfev=10000)
                fit_x = np.linspace(param_values.min(), param_values.max(), 100)
                ax.plot(fit_x, _exp_decay(fit_x, *popt), '-', color='#2980b9',
                       linewidth=1.5, alpha=0.5, zorder=2)
            except (RuntimeError, ValueError):
                pass
        
        display_name = display_names.get(param_name, param_name)
        ax.set_ylabel('Fidelity', fontsize=9)
        ax.set_title(display_name, fontsize=10, fontweight='bold', pad=8)
        ax.set_ylim([0, 1.05])
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Format x-axis with proper scientific notation
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True)
        ax.tick_params(axis='both', labelsize=8)
        
        # Only show legend on first param plot to avoid clutter
        if i == 0:
            ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
    
    # Add overall title
    model_suffix = " (TwoZone)" if _USE_TWO_ZONE else " (OneZone)"
    fig.suptitle(f'd=3 [[7,1,3]] vs d=5 [[17,1,5]] Noise Comparison{model_suffix}', 
                fontsize=14, fontweight='bold', y=0.98)
    
    # Adjust layout to prevent overlapping
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution routine for d=3 vs d=5 comparison analysis."""
    global _USE_TWO_ZONE
    
    print("\n" + "="*80)
    print("d=3 vs d=5 DISTILLATION FIDELITY COMPARISON")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Model selection
    print("\nSelect noise model:")
    print("  1. GeminiOneZoneNoiseModel (3 params: cz_unpaired)")
    print("  2. GeminiTwoZoneNoiseModel (9 params: cz_unpaired + mover + sitter)")
    choice = input("Enter choice [1/2]: ").strip()
    _USE_TWO_ZONE = (choice == "2")
    
    print(f"\n✓ Using: {get_model_name()}")
    
    # Configuration
    SCALING_COEFFS = np.linspace(0.1, 3.0, 25)
    PARAM_ITERATIONS = 15
    SHOTS_PER_TEST = 500
    
    noise_params = get_noise_params()
    # 2 variants (d3, d5) × (scaling + param sweeps)
    total_tests = 2 * (len(SCALING_COEFFS) + len(noise_params) * PARAM_ITERATIONS)
    
    print(f"\nConfiguration:")
    print(f"  - Circuits compared:")
    print(f"      d=3 [[7,1,3]] injection (7 qubits)")
    print(f"      d=5 [[17,1,5]] injection (17 qubits)")
    print(f"  - Noise model: {get_model_name()}")
    print(f"  - Scaling coefficients: {len(SCALING_COEFFS)} points")
    print(f"  - Parameters to test: {noise_params}")
    print(f"  - Iterations per parameter: {PARAM_ITERATIONS}")
    print(f"  - Shots per test: {SHOTS_PER_TEST}")
    print(f"  - Parallel workers: {NUM_WORKERS}")
    print(f"  - Total simulations: {total_tests}")
    
    # ---- ANALYSIS 1: Scaling Comparison ----
    print("\n" + "-"*80)
    print("ANALYSIS 1: Scaling Coefficient Comparison (d=3 vs d=5)")
    print("-"*80)
    
    scaling_results = analyze_scaling_comparison(
        SCALING_COEFFS, shots=SHOTS_PER_TEST, verbose=True
    )
    
    print(f"\n  Scaling Summary:")
    print(f"    d=3: min={scaling_results['d3'].min():.4f}, max={scaling_results['d3'].max():.4f}, mean={scaling_results['d3'].mean():.4f}")
    print(f"    d=5: min={scaling_results['d5'].min():.4f}, max={scaling_results['d5'].max():.4f}, mean={scaling_results['d5'].mean():.4f}")
    
    # ---- ANALYSIS 2: Parameter Sweep Comparison ----
    print("\n" + "-"*80)
    print("ANALYSIS 2: Parameter Sweep Comparison (d=3 vs d=5)")
    print("-"*80)
    
    param_results = analyze_params_comparison(
        iterations=PARAM_ITERATIONS, shots=SHOTS_PER_TEST, verbose=True
    )
    
    print("\n  Parameter Comparison (fidelity drop):")
    for param_name, data in param_results.items():
        d3_drop = data['d3'][0] - data['d3'][-1]
        d5_drop = data['d5'][0] - data['d5'][-1]
        print(f"    {param_name}: d3={d3_drop:+.4f}, d5={d5_drop:+.4f}")
    
    # ---- VISUALIZATION ----
    print("\n" + "-"*80)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("-"*80)
    
    fig = plot_comparison_results(scaling_results, param_results)
    
    model_suffix = "two_zone" if _USE_TWO_ZONE else "one_zone"
    png_filename = f'd3_vs_d5_comparison_{model_suffix}.png'
    json_filename = f'd3_vs_d5_comparison_{model_suffix}_results.json'
    
    fig.savefig(png_filename, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {png_filename}")
    
    # ---- SAVE RESULTS ----
    print("\n" + "-"*80)
    print("SAVING RESULTS")
    print("-"*80)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'comparison': 'd3_vs_d5',
        'circuits': {
            'd3': '[[7,1,3]]_injection (7 qubits)',
            'd5': '[[17,1,5]]_injection (17 qubits)',
        },
        'noise_model': get_model_name(),
        'configuration': {
            'scaling_coefficients': len(SCALING_COEFFS),
            'param_iterations': PARAM_ITERATIONS,
            'shots_per_test': SHOTS_PER_TEST,
            'parameters_tested': noise_params,
            'total_simulations': total_tests,
        },
        'scaling_analysis': {
            'coefficients': scaling_results['coeffs'].tolist(),
            'd3_fidelities': scaling_results['d3'].tolist(),
            'd5_fidelities': scaling_results['d5'].tolist(),
        },
        'parameter_analyses': {
            param_name: {
                'values': data['values'].tolist(),
                'd3_fidelities': data['d3'].tolist(),
                'd5_fidelities': data['d5'].tolist(),
                'd3_fidelity_drop': float(data['d3'][0] - data['d3'][-1]),
                'd5_fidelity_drop': float(data['d5'][0] - data['d5'][-1]),
            }
            for param_name, data in param_results.items()
        },
    }
    
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved: {json_filename}")
    
    # ---- SUMMARY ----
    print("\n" + "="*80)
    print("COMPARISON ANALYSIS COMPLETE")
    print("="*80)
    
    print(f"\n📊 d=3 [[7,1,3]] (7 qubits) Results:")
    print(f"    Scaling mean fidelity: {scaling_results['d3'].mean():.4f}")
    print(f"    Scaling fidelity range: [{scaling_results['d3'].min():.4f}, {scaling_results['d3'].max():.4f}]")
    
    print(f"\n📊 d=5 [[17,1,5]] (17 qubits) Results:")
    print(f"    Scaling mean fidelity: {scaling_results['d5'].mean():.4f}")
    print(f"    Scaling fidelity range: [{scaling_results['d5'].min():.4f}, {scaling_results['d5'].max():.4f}]")
    
    # Comparison summary
    d3_avg = scaling_results['d3'].mean()
    d5_avg = scaling_results['d5'].mean()
    diff = d5_avg - d3_avg
    print(f"\n📈 Comparison:")
    print(f"    Mean fidelity difference (d5 - d3): {diff:+.4f}")
    if diff > 0:
        print(f"    → d=5 [[17,1,5]] shows HIGHER average fidelity")
    elif diff < 0:
        print(f"    → d=3 [[7,1,3]] shows HIGHER average fidelity")
    else:
        print(f"    → Both circuits show EQUAL average fidelity")
    
    # Find most sensitive parameter for each
    d3_sens = {name: abs(data['d3'][0] - data['d3'][-1]) for name, data in param_results.items()}
    d5_sens = {name: abs(data['d5'][0] - data['d5'][-1]) for name, data in param_results.items()}
    d3_most = max(d3_sens, key=d3_sens.get)
    d5_most = max(d5_sens, key=d5_sens.get)
    print(f"\n    Most sensitive param (d3): {d3_most} (drop={d3_sens[d3_most]:.4f})")
    print(f"    Most sensitive param (d5): {d5_most} (drop={d5_sens[d5_most]:.4f})")
    
    plt.show()
    
    return results


if __name__ == "__main__":
    results = main()
