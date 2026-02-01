"""
magictesting.py - Magic State Preparation Comparison Analysis
==============================================================

This module compares two magic state preparation methods:
1. Original: T gate only
2. Modified: H gate followed by T gate

Tests scaling coefficient and CZ unpaired gate noise parameters.
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

Slot = Tuple[int, int]  # (tuple index, position inside tuple)
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
    result = list(target)  # Make a copy to avoid mutating input
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
        # Check if all angles are the same (global gate) - must have angles to compare
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
            swaps, intsc_rev, move_pauli_channel, sitter_pauli_channel, nqubs
        )
        dumb_circ.append(swap_noise_circ)

    return dumb_circ


# ============================================================================
# CUSTOM TWO-ZONE NOISE MODEL (from a3_twozone.ipynb)
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


def get_model_name():
    """Get the display name of the current model."""
    return "CustomGeminiTwoZoneNoiseModel" if _USE_TWO_ZONE else "GeminiOneZoneNoiseModel"


# ============================================================================
# MAGIC STATE PREPARATION VARIANTS
# ============================================================================

@squin.kernel
def magicstateprep_t_only(qubits, ind):
    """Original magic state prep: T gate only."""
    squin.t(qubits[ind])


@squin.kernel
def magicstateprep_h_then_t(qubits, ind):
    """Modified magic state prep: H gate before T gate."""
    squin.h(qubits[ind])
    squin.t(qubits[ind])


# ============================================================================
# INJECTION CIRCUITS (two variants)
# ============================================================================

@squin.kernel
def injection_t_only(q: ilist.IList[Qubit, Literal[7]]):
    """Injection using T-only magic state prep."""
    squin.reset(q[0:2])
    squin.reset(q[3:7])
    magicstateprep_t_only(q, 2)
    
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
def injection_h_then_t(q: ilist.IList[Qubit, Literal[7]]):
    """Injection using H+T magic state prep."""
    squin.reset(q[0:2])
    squin.reset(q[3:7])
    magicstateprep_h_then_t(q, 2)
    
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


# ============================================================================
# STEANE ENCODING
# ============================================================================

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


# ============================================================================
# A3 CIRCUITS (two variants)
# ============================================================================

@squin.kernel
def a3_circuit_t_only():
    """A3 circuit with T-only magic state prep."""
    q = squin.qalloc(21)
    injection_t_only(q[0:7])
    
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


@squin.kernel
def a3_circuit_h_then_t():
    """A3 circuit with H+T magic state prep."""
    q = squin.qalloc(21)
    injection_h_then_t(q[0:7])
    
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
# FIDELITY AND SYNDROME DETECTION
# ============================================================================

SYND_INDICES = [[1, 3, 5, 7], [4, 5, 6, 7], [2, 3, 6, 7]]


def _check_syndrome(bits, indices):
    """Check if syndrome is triggered (product of parities == 1)."""
    parity = np.prod(1 - 2 * bits[np.array(indices) - 1])
    return parity == 1


def find_good_rate(samples):
    """Calculate fidelity: fraction of samples with all 6 syndromes triggered."""
    samples = np.asarray(samples)
    good = 0
    for sample in samples:
        x_bits, z_bits = sample[:7], sample[7:14]
        count = sum(_check_syndrome(x_bits, idx) for idx in SYND_INDICES)
        count += sum(_check_syndrome(z_bits, idx) for idx in SYND_INDICES)
        good += (count == 6)
    return good / len(samples)


# ============================================================================
# SIMULATION ENGINE
# ============================================================================

# Circuit caches for both variants
_circuit_cache = {
    't_only': None,
    'h_then_t': None
}


def _get_circuit(variant):
    """Get cached circuit for specified variant."""
    global _circuit_cache
    if _circuit_cache[variant] is None:
        if variant == 't_only':
            _circuit_cache[variant] = emit_circuit(a3_circuit_t_only)
        else:
            _circuit_cache[variant] = emit_circuit(a3_circuit_h_then_t)
    return _circuit_cache[variant]


def run_simulation(variant, noise_model, shots=1000):
    """Run circuit variant with noise model and return fidelity."""
    if _USE_TWO_ZONE:
        cirq_circuit = _transform_circuit_two_zone(_get_circuit(variant), model=noise_model)
    else:
        cirq_circuit = noise.transform_circuit(_get_circuit(variant), model=noise_model)
    stim_circuit = bloqade.stim.Circuit(load_circuit(cirq_circuit))
    samples = np.array(stim_circuit.compile_sampler().sample(shots=shots))
    return find_good_rate(samples)


# ============================================================================
# NOISE PARAMETERS TO TEST
# ============================================================================

# Base noise parameters
_BASE_NOISE_PARAMS = [
    'cz_unpaired_gate_px',
    'cz_unpaired_gate_py',
    'cz_unpaired_gate_pz',
]

# Additional parameters only for GeminiTwoZoneNoiseModel
_TWO_ZONE_EXTRA_PARAMS = [
    'sitter_px', 'sitter_py', 'sitter_pz',
    'mover_px', 'mover_py', 'mover_pz',
]


def get_noise_params():
    """Get the appropriate noise parameters based on selected model."""
    if _USE_TWO_ZONE:
        return _BASE_NOISE_PARAMS + _TWO_ZONE_EXTRA_PARAMS
    return _BASE_NOISE_PARAMS


def get_zero_params():
    """Get zero params dict for the selected model."""
    return {p: 0.0 for p in get_noise_params()}


# ============================================================================
# PARALLEL TEST WORKERS
# ============================================================================

def _run_scaling_test(args):
    """Worker for scaling coefficient test."""
    variant, coeff, shots = args
    NoiseModelClass = get_noise_model_class()
    noise_model = NoiseModelClass(scaling_factor=coeff)
    fidelity = run_simulation(variant, noise_model, shots=shots)
    return (variant, coeff), fidelity


def _run_param_test(args):
    """Worker for parameter sweep test."""
    variant, param_name, param_val, shots = args
    params = get_zero_params()
    params[param_name] = param_val
    NoiseModelClass = get_noise_model_class()
    noise_model = NoiseModelClass(**params)
    fidelity = run_simulation(variant, noise_model, shots=shots)
    return (variant, param_name, param_val), fidelity


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_scaling_comparison(coeff_range, shots=500, verbose=True):
    """Compare scaling coefficient behavior between both variants."""
    all_args = []
    for variant in ['t_only', 'h_then_t']:
        for coeff in coeff_range:
            all_args.append((variant, coeff, shots))
    
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
    
    # Organize results by variant
    t_only_fids = np.array([results[('t_only', c)] for c in coeff_range])
    h_then_t_fids = np.array([results[('h_then_t', c)] for c in coeff_range])
    
    return {
        't_only': t_only_fids,
        'h_then_t': h_then_t_fids
    }


def analyze_params_comparison(iterations=20, shots=500, verbose=True):
    """Compare parameter sweeps between both variants."""
    all_args = []
    param_ranges = {}
    
    noise_params = get_noise_params()
    
    for param_name in noise_params:
        # gate params and mover/sitter get larger range
        if 'gate' in param_name.lower() or 'mover' in param_name.lower() or 'sitter' in param_name.lower():
            param_values = np.linspace(0, 5e-2, iterations)
        else:
            param_values = np.linspace(0, 2e-3, iterations)
        param_ranges[param_name] = param_values
        for variant in ['t_only', 'h_then_t']:
            for pv in param_values:
                all_args.append((variant, param_name, pv, shots))
    
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
        t_only_fids = np.array([results[('t_only', param_name, pv)] for pv in pvals])
        h_then_t_fids = np.array([results[('h_then_t', param_name, pv)] for pv in pvals])
        param_results[param_name] = {
            'values': pvals,
            't_only': t_only_fids,
            'h_then_t': h_then_t_fids
        }
    
    return param_results


# ============================================================================
# VISUALIZATION
# ============================================================================

# Fit functions
def _exp_decay(x, A, b):
    return A * np.exp(-b * x)


def _r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0


def plot_comparison(coeffs, scaling_results, param_results=None):
    """Create comparison plot for scaling factor only."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color scheme
    colors = {
        't_only': '#e74c3c',      # Red
        'h_then_t': '#3498db'     # Blue
    }
    labels = {
        't_only': 'Non-Magic',
        'h_then_t': 'Magic'
    }
    
    for variant in ['t_only', 'h_then_t']:
        fids = scaling_results[variant]
        ax.scatter(coeffs, fids, s=60, alpha=0.7, color=colors[variant],
                  edgecolors='black', linewidth=0.5, label=f'{labels[variant]} (data)')
        
        # Fit exponential decay
        try:
            popt, _ = curve_fit(_exp_decay, coeffs, fids, p0=[fids[0], 0.5], maxfev=10000)
            fit_x = np.linspace(coeffs.min(), coeffs.max(), 100)
            r2 = _r_squared(fids, _exp_decay(coeffs, *popt))
            ax.plot(fit_x, _exp_decay(fit_x, *popt), '--', color=colors[variant],
                   linewidth=2, alpha=0.8, 
                   label=f'{labels[variant]}: {popt[0]:.3f}·e^(-{popt[1]:.3f}c) [R²={r2:.3f}]')
        except (RuntimeError, ValueError):
            pass
    
    ax.set_xlabel('Scaling Coefficient', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fidelity', fontsize=12, fontweight='bold')
    model_suffix = " (TwoZone)" if _USE_TWO_ZONE else " (OneZone)"
    ax.set_title(f'Scaling Factor: Non-Magic vs Magic{model_suffix}', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution routine for magic state prep comparison."""
    global _USE_TWO_ZONE
    
    print("\n" + "="*80)
    print("MAGIC STATE PREPARATION COMPARISON")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Model selection
    print("\nSelect noise model:")
    print("  1. GeminiOneZoneNoiseModel (default)")
    print("  2. GeminiTwoZoneNoiseModel (custom)")
    choice = input("Enter choice [1/2]: ").strip()
    _USE_TWO_ZONE = (choice == "2")
    
    print(f"\n✓ Using: {get_model_name()}")
    
    print("\nComparing:")
    print("  • Non-Magic:  magicstateprep applies T gate only")
    print("  • Magic:      magicstateprep applies H then T gate")
    
    # Configuration
    SCALING_COEFFS = np.linspace(0.1, 3.0, 25)
    PARAM_ITERATIONS = 15
    SHOTS_PER_TEST = 500
    
    noise_params = get_noise_params()
    total_tests = 2 * len(SCALING_COEFFS) + 2 * len(noise_params) * PARAM_ITERATIONS
    
    print(f"\nConfiguration:")
    print(f"  - Noise model: {get_model_name()}")
    print(f"  - Scaling coefficients: {len(SCALING_COEFFS)} points × 2 variants")
    print(f"  - Parameters to test: {noise_params}")
    print(f"  - Iterations per parameter: {PARAM_ITERATIONS} × 2 variants")
    print(f"  - Shots per test: {SHOTS_PER_TEST}")
    print(f"  - Parallel workers: {NUM_WORKERS}")
    print(f"  - Total simulations: {total_tests}")
    
    # ---- ANALYSIS 1: Scaling Coefficient Comparison ----
    print("\n" + "-"*80)
    print("ANALYSIS 1: Scaling Coefficient Comparison")
    print("-"*80)
    
    scaling_results = analyze_scaling_comparison(SCALING_COEFFS, shots=SHOTS_PER_TEST, verbose=True)
    
    print("\n  Scaling Summary:")
    for variant in ['t_only', 'h_then_t']:
        fids = scaling_results[variant]
        print(f"    {variant:10s}: min={fids.min():.4f}, max={fids.max():.4f}, mean={fids.mean():.4f}")
    
    # ---- ANALYSIS 2: Parameter Sweeps Comparison ----
    print("\n" + "-"*80)
    print("ANALYSIS 2: CZ Unpaired Gate Parameter Comparison")
    print("-"*80)
    
    param_results = analyze_params_comparison(iterations=PARAM_ITERATIONS, shots=SHOTS_PER_TEST, verbose=True)
    
    print("\n  Parameter Summaries:")
    for param_name in noise_params:
        data = param_results[param_name]
        print(f"    {param_name}:")
        for variant in ['t_only', 'h_then_t']:
            fids = data[variant]
            drop = fids[0] - fids[-1]
            print(f"      {variant:10s}: drop={drop:+.4f}")
    
    # ---- VISUALIZATION ----
    print("\n" + "-"*80)
    print("GENERATING VISUALIZATIONS")
    print("-"*80)
    
    fig = plot_comparison(SCALING_COEFFS, scaling_results, param_results)
    
    # Model-specific filenames
    model_suffix = "two_zone" if _USE_TWO_ZONE else "one_zone"
    png_filename = f'magic_state_comparison_{model_suffix}.png'
    json_filename = f'magic_state_comparison_{model_suffix}_results.json'
    
    fig.savefig(png_filename, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {png_filename}")
    
    # ---- SAVE RESULTS ----
    print("\n" + "-"*80)
    print("SAVING RESULTS")
    print("-"*80)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'noise_model': get_model_name(),
        'configuration': {
            'scaling_coefficients': len(SCALING_COEFFS),
            'param_iterations': PARAM_ITERATIONS,
            'shots_per_test': SHOTS_PER_TEST,
            'parameters_tested': noise_params,
            'total_simulations': total_tests,
        },
        'variants': {
            't_only': 'Non-Magic: T gate only',
            'h_then_t': 'Magic: H gate then T gate'
        },
        'scaling_analysis': {
            'coefficients': SCALING_COEFFS.tolist(),
            't_only_fidelities': scaling_results['t_only'].tolist(),
            'h_then_t_fidelities': scaling_results['h_then_t'].tolist(),
        },
        'parameter_analyses': {
            param_name: {
                'values': data['values'].tolist(),
                't_only_fidelities': data['t_only'].tolist(),
                'h_then_t_fidelities': data['h_then_t'].tolist(),
                't_only_drop': float(data['t_only'][0] - data['t_only'][-1]),
                'h_then_t_drop': float(data['h_then_t'][0] - data['h_then_t'][-1]),
            }
            for param_name, data in param_results.items()
        },
    }
    
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved: {json_filename}")
    
    # ---- SUMMARY ----
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    # Determine winner
    t_only_mean = scaling_results['t_only'].mean()
    h_then_t_mean = scaling_results['h_then_t'].mean()
    
    print(f"\n📊 Comparison Results:")
    print(f"    Non-Magic mean fidelity:  {t_only_mean:.4f}")
    print(f"    Magic mean fidelity:      {h_then_t_mean:.4f}")
    
    if t_only_mean > h_then_t_mean:
        print(f"\n✅ Non-Magic performs better by {(t_only_mean - h_then_t_mean)*100:.2f}%")
    elif h_then_t_mean > t_only_mean:
        print(f"\n✅ Magic performs better by {(h_then_t_mean - t_only_mean)*100:.2f}%")
    else:
        print(f"\n🔄 Both methods perform equally")
    
    plt.show()
    
    return results


if __name__ == "__main__":
    results = main()
