"""
noisetesting.py - Comprehensive Noise Analysis for A3 Circuit, parts include code attributed to bloqade-circuit (https://github.com/QuEraComputing/bloqade-circuit/tree/main). Apache License 2.0 with LLVM Exceptions. Modified by Team Piqasso, 1 Feb. 2026.
==============================================================

This module performs advanced noise testing on the A3 quantum circuit,
combining simulations from a3.ipynb with customized noise models from emiliano.ipynb.

Features:
- Fidelity calculations using Steane code syndrome detection
- Scaling coefficient analysis (GeminiOneZoneNoiseModel)
- Custom parameter iteration analysis (GeminiOneZoneNoiseModel parameters)
- Comprehensive visualization and plotting
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

# Optimize for Snapdragon X Elite - use all available cores
NUM_WORKERS = min(os.cpu_count() or 4, 10)  # Cap at 10 to avoid overhead


# ============================================================================
# TWO-ZONE NOISE MODEL HELPER FUNCTIONS (from a3_twozone.ipynb)
# ============================================================================

Slot = Tuple[int, int]  # (tuple index, position inside tuple)
Swap = Tuple[Slot, Slot]

## MODIFICATION: COPIED FROM BLOQADE CIRCUIT(https://github.com/QuEraComputing/bloqade-circuit/tree/main), MODIFICATIONSADDED BY TEAM PIQASSO, NOT IN ORIGINAL REPOSITORY
## MODIFICATION: PRINT STATEMENTS FOR UNIT TESTING HAVE BEEN ADDED
## DISCLAIMER: COPIED FROM bloqade-circuit/src/circ_utils/noise/_two_zone_utils.py, model.py, transform.py
def get_equivalent_swaps_greedy(source, target): ##MODIFICATION: ADDED BY TEAM PIQASSO, NOT IN ORIGINAL REPOSITORY
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
        new_moment, first_move_added = _add_move_and_sitter_channels(
            prev_qargs,
            curr_qargs,
            new_moment,
            qub_reg,
            sitter_pauli_channel,
            move_pauli_channel,
        )
        dumb_circ.append(new_moment)

        new_moment = cirq.Moment()
        new_moment, second_move_added = _add_move_and_sitter_channels(
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

        swaps = get_equivalent_swaps_greedy(
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


def transform_circuit_two_zone(
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
# CIRCUIT DEFINITIONS (from a3.ipynb)
# ============================================================================

@squin.kernel
def magicstateprep(qubits, ind):
    #squin.h(qubits[ind])
    squin.t(qubits[ind])


@squin.kernel
def injection(q: ilist.IList[Qubit, Literal[7]]):
    """Apply magic-state injection to the 7-qubit register `q` (allocated by caller)."""
    squin.reset(q[0:2])
    squin.reset(q[3:7])
    magicstateprep(q, 2)
    # ry(-pi/2) on old 0..5  ->  new [3,1,0,6,4,5]
    for j in (3, 1, 0, 6, 4, 5):
        squin.ry(-pi / 2, q[j])

    # cz(1,2), cz(3,4), cz(5,6) -> cz(1,0), cz(6,4), cz(5,2)
    squin.cz(q[1], q[0])
    squin.cz(q[6], q[4])
    squin.cz(q[5], q[2])

    # ry on old 6 -> new 2
    squin.ry(pi / 2, q[2])

    # cz(0,3), cz(2,5), cz(4,6) -> cz(3,6), cz(0,5), cz(4,2)
    squin.cz(q[3], q[6])
    squin.cz(q[0], q[5])
    squin.cz(q[4], q[2])

    # ry on old 2..6 -> new [0,6,4,5,2]
    for j in (0, 6, 4, 5, 2):
        squin.ry(pi / 2, q[j])

    # cz(0,1), cz(2,3), cz(4,5) -> cz(3,1), cz(0,6), cz(4,5)
    squin.cz(q[3], q[1])
    squin.cz(q[0], q[6])
    squin.cz(q[4], q[5])

    # final single-qubit ry: old 1 -> 1, old 2 -> 0, old 4 -> 4
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
def a3_circuit():
    """A3 circuit implementation in squin - Fault-tolerant Steane syndrome extraction."""
    q = squin.qalloc(21)
    
    #steane_encode_zero_on(q[0:7])
    injection(q[0:7])
    # Qubits 0-6: logical data
    # Qubits 7-13 + 14-20: ancilla blocks

    # Prepare ancilla as |+⟩_L for first half (X-stabilizer syndrome)
    steane_encode_plus_on(q[7:14])

    # CNOTs: data → ancilla (X-syndrome extraction)
    for i in range(7):
        squin.cx(q[i], q[i+7])
    
    # Prepare ancilla as |0⟩_L for second half (Z-stabilizer syndrome)
    steane_encode_zero_on(q[14:21])

    # CNOTs: ancilla → data (Z-syndrome extraction)
    for i in range(7):
        squin.cx(q[i+14], q[i])

    # Transversal H on ancilla
    for i in range(7):
        squin.h(q[i+14])
    
    # Measure ancilla qubits 7-21 (Syndrome bits)
    for i in range(7, 21):
        squin.measure(q[i])


# ============================================================================
# FIDELITY AND SYNDROME DETECTION
# ============================================================================

# Syndrome indices for Steane [[7,1,3]] code
SYND_INDICES = [[1, 3, 5, 7], [4, 5, 6, 7], [2, 3, 6, 7]]

# Global circuit cache
_cached_circuit = None


def _check_syndrome(bits, indices):
    """Check if syndrome is triggered (product of parities == 1)."""
    # Convert bits to ±1 and compute product
    parity = np.prod(1 - 2 * bits[np.array(indices) - 1])
    return parity == 1


def find_good_rate(samples):
    """Calculate fidelity: fraction of samples with all 6 syndromes triggered."""
    samples = np.asarray(samples)
    good = 0
    for sample in samples:
        x_bits, z_bits = sample[:7], sample[7:14]
        # Count triggered syndromes (need all 6)
        count = sum(_check_syndrome(x_bits, idx) for idx in SYND_INDICES)
        count += sum(_check_syndrome(z_bits, idx) for idx in SYND_INDICES)
        good += (count == 6)
    return good / len(samples)


# ============================================================================
# SIMULATION ENGINE
# ============================================================================

def _get_circuit():
    """Get cached circuit, generating if needed."""
    global _cached_circuit
    if _cached_circuit is None:
        _cached_circuit = emit_circuit(a3_circuit)
    return _cached_circuit


def run_simulation_with_noise(noise_model, shots=1000):
    """Run A3 circuit with noise model and return fidelity."""
    if _USE_TWO_ZONE:
        cirq_circuit = transform_circuit_two_zone(_get_circuit(), model=noise_model)
    else:
        cirq_circuit = noise.transform_circuit(_get_circuit(), model=noise_model)
    stim_circuit = bloqade.stim.Circuit(load_circuit(cirq_circuit))
    samples = np.array(stim_circuit.compile_sampler().sample(shots=shots))
    return find_good_rate(samples)


# ============================================================================
# ANALYSIS 1: Scaling Coefficient Analysis
# ============================================================================

def _run_scaling_test(args):
    """Worker function for parallel scaling coefficient tests."""
    coeff, shots = args
    NoiseModelClass = get_noise_model_class()
    noise_model = NoiseModelClass(scaling_factor=coeff)
    return coeff, run_simulation_with_noise(noise_model, shots=shots)


def _run_parallel(worker_fn, args_list, key_fn, verbose, label):
    """Generic parallel executor with progress output."""
    total = len(args_list)
    results = {}
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(worker_fn, args): key_fn(args) for args in args_list}
        for i, future in enumerate(as_completed(futures), 1):
            key, val = future.result()
            results[key] = val
            if verbose:
                print(f"  [{i:2d}/{total}] {label(key, val)}")
    return results


def analyze_scaling_coefficients(coeff_range=None, shots=500, verbose=True):
    """Test fidelity across GeminiOneZoneNoiseModel scaling factors."""
    if coeff_range is None:
        coeff_range = np.arange(0.5, 1.5, 0.1)
    
    if verbose:
        print(f"  Running {len(coeff_range)} tests in parallel ({NUM_WORKERS} workers)...")
    
    args_list = [(c, shots) for c in coeff_range]
    results = _run_parallel(
        _run_scaling_test, args_list,
        key_fn=lambda a: a[0],
        verbose=verbose,
        label=lambda c, f: f"coeff={c:.2f} → Fidelity: {f:.4f}"
    )
    
    fidelities = [results[c] for c in coeff_range]
    return np.array(coeff_range), np.array(fidelities)


# ============================================================================
# ANALYSIS 2: Custom Noise Parameter Iteration
# ============================================================================

# Base noise parameters (shared by both OneZone and TwoZone models)
_BASE_NOISE_PARAMS = [
    'global_px', 'global_py', 'global_pz',
    'local_px', 'local_py', 'local_pz',
    'local_unaddressed_px', 'local_unaddressed_py', 'local_unaddressed_pz',
    'cz_paired_gate_px', 'cz_paired_gate_py', 'cz_paired_gate_pz',
    'cz_unpaired_gate_px', 'cz_unpaired_gate_py', 'cz_unpaired_gate_pz',
]

# Additional parameters only for GeminiTwoZoneNoiseModel
_TWO_ZONE_EXTRA_PARAMS = [
    'sitter_px', 'sitter_py', 'sitter_pz',
    'mover_px', 'mover_py', 'mover_pz',
]

# Will be set based on user selection
_NOISE_PARAMS = _BASE_NOISE_PARAMS.copy()
_USE_TWO_ZONE = False


def get_noise_params():
    """Get the appropriate noise parameters based on selected model."""
    if _USE_TWO_ZONE:
        return _BASE_NOISE_PARAMS + _TWO_ZONE_EXTRA_PARAMS
    return _BASE_NOISE_PARAMS


def get_zero_params():
    """Get zero params dict for the selected model."""
    return {p: 0.0 for p in get_noise_params()}


def get_noise_model_class():
    """Get the appropriate noise model class."""
    if _USE_TWO_ZONE:
        return CustomGeminiTwoZoneNoiseModel
    return noise.GeminiOneZoneNoiseModel


def get_model_name():
    """Get the display name of the current model."""
    return "CustomGeminiTwoZoneNoiseModel" if _USE_TWO_ZONE else "GeminiOneZoneNoiseModel"

def _run_custom_param_test(args):
    """Worker function for parallel custom parameter tests."""
    param_name, param_val, shots = args
    params = get_zero_params()
    params[param_name] = param_val
    NoiseModelClass = get_noise_model_class()
    noise_model = NoiseModelClass(**params)
    return (param_name, param_val), run_simulation_with_noise(noise_model, shots=shots)


def analyze_all_parameters_batched(iterations=20, shots=500, verbose=True):
    """
    Run ALL parameter sweeps in a SINGLE parallel pool (much faster).
    Returns dict: {param_name: (param_values, fidelities)}
    """
    # Build all args for all parameters at once
    all_args = []
    param_ranges = {}
    
    noise_params = get_noise_params()
    
    for param_name in noise_params:
        # gate params get larger range; mover/sitter also get larger range
        if 'gate' in param_name.lower() or 'mover' in param_name.lower() or 'sitter' in param_name.lower():
            prange = (0, 5e-2)
        else:
            prange = (0, 2e-3)
        param_values = np.linspace(prange[0], prange[1], iterations)
        param_ranges[param_name] = param_values
        for pv in param_values:
            all_args.append((param_name, pv, shots))
    
    total = len(all_args)
    if verbose:
        print(f"  Running {total} tests in single parallel pool ({NUM_WORKERS} workers)...")
    
    # Single parallel execution for ALL tests
    results = {}  # (param_name, param_val) -> fidelity
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(_run_custom_param_test, args): args for args in all_args}
        for i, future in enumerate(as_completed(futures), 1):
            key, fidelity = future.result()
            results[key] = fidelity
            if verbose and i % 50 == 0:
                print(f"    Progress: {i}/{total} ({100*i/total:.0f}%)")
    
    if verbose:
        print(f"    Progress: {total}/{total} (100%)")
    
    # Reorganize into per-parameter results
    all_param_results = {}
    for param_name in noise_params:
        pvals = param_ranges[param_name]
        fids = np.array([results[(param_name, pv)] for pv in pvals])
        all_param_results[param_name] = (pvals, fids)
    
    return all_param_results


# ============================================================================
# VISUALIZATION
# ============================================================================

# Fit functions
def _exp_decay(x, A, b): return A * np.exp(-b * x)
def _power_law(x, A, n): return A * np.power(x, -n)
def _logarithmic(x, A, b): return A - b * np.log(x)
def _r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0


def plot_individual_analyses(coeffs, scaling_fidelities, all_param_results):
    """Create detailed plots: scaling analysis + grid of all parameter sweeps.
    
    Args:
        coeffs: Array of scaling coefficients tested
        scaling_fidelities: Array of fidelities for each scaling coefficient
        all_param_results: Either dict {param_name: (values, fidelities)} or 
                          list of (param_name, values, fidelities) tuples
    """
    n_params = len(all_param_results)
    n_cols = 4
    n_rows = (n_params + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure with scaling plot on top, parameter grid below
    fig = plt.figure(figsize=(16, 5 + 4 * n_rows))
    
    # ---- Top Plot: Scaling Analysis with Multiple Curve Fits ----
    ax_top = fig.add_subplot(n_rows + 1, 1, 1)
    ax_top.scatter(coeffs, scaling_fidelities, s=80, alpha=0.8, color='#2980b9', 
                   edgecolors='black', linewidth=1, zorder=5, label='Measured')
    
    fit_x = np.linspace(coeffs.min(), coeffs.max(), 100)
    mask = coeffs > 0
    fit_results = []
    
    fits = [
        ('Exp', _exp_decay, [scaling_fidelities[0], 0.5], ([0, 0], [2, 10]), '-', '#e74c3c',
         lambda p: f'{p[0]:.3f}·e^(-{p[1]:.3f}c)', coeffs, scaling_fidelities),
        ('Power', _power_law, [scaling_fidelities[0], 0.5], ([0, -5], [10, 5]), '--', '#27ae60',
         lambda p: f'{p[0]:.3f}·c^({-p[1]:.3f})', coeffs[mask], scaling_fidelities[mask]),
        ('Log', _logarithmic, [scaling_fidelities.mean(), 0.1], None, '-.', '#9b59b6',
         lambda p: f'{p[0]:.3f}-{p[1]:.3f}·ln(c)', coeffs[mask], scaling_fidelities[mask]),
    ]
    
    for name, func, p0, bounds, ls, color, label_fn, x_data, y_data in fits:
        try:
            kwargs = {'p0': p0, 'maxfev': 10000}
            if bounds:
                kwargs['bounds'] = bounds
            popt, _ = curve_fit(func, x_data, y_data, **kwargs)
            r2 = _r_squared(y_data, func(x_data, *popt))
            fit_results.append((name, r2))
            ax_top.plot(fit_x, func(fit_x, *popt), ls, color=color, linewidth=2,
                       alpha=0.8, label=f'{name}: {label_fn(popt)} [R²={r2:.3f}]', zorder=4)
        except (RuntimeError, ValueError, TypeError):
            pass  # Curve fit failed, skip this fit type
    
    if fit_results:
        best = max(fit_results, key=lambda x: x[1])
        print(f"  Best fit for scaling analysis: {best[0]} (R²={best[1]:.4f})")
    
    ax_top.set_xlabel('Scaling Coefficient (c)', fontsize=10, fontweight='bold')
    ax_top.set_ylabel('Fidelity', fontsize=10, fontweight='bold')
    ax_top.set_title('Fidelity Decay vs Scaling Coefficient (Exp / Power / Log Fits)', fontsize=11, fontweight='bold')
    ax_top.set_ylim([0, 1.05])
    ax_top.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax_top.legend(fontsize=7, loc='best')
    
    # ---- Grid of Parameter Sweeps ----
    colors = ['#e74c3c', '#3498db', '#27ae60', '#f39c12', '#9b59b6',
              '#1abc9c', '#e67e22', '#34495e', '#16a085', '#c0392b',
              '#2980b9', '#8e44ad', '#d35400', '#2c3e50', '#7f8c8d']
    
    # Handle both dict and list of tuples input
    if isinstance(all_param_results, dict):
        param_items = [(name, *data) for name, data in all_param_results.items()]
    else:
        param_items = all_param_results
    
    for i, (param_name, param_values, fidelities) in enumerate(param_items):
        ax = fig.add_subplot(n_rows + 1, n_cols, n_cols + 1 + i)
        color = colors[i % len(colors)]
        
        ax.scatter(param_values, fidelities, s=40, alpha=0.7, color=color,
                   edgecolors='black', linewidth=0.5, zorder=3)
        
        # Fit exponential decay
        if np.std(fidelities) > 0.01:
            try:
                popt, _ = curve_fit(_exp_decay, param_values, fidelities,
                                   p0=[fidelities[0], 100], maxfev=10000)
                fit_x = np.linspace(param_values.min(), param_values.max(), 100)
                ax.plot(fit_x, _exp_decay(fit_x, *popt), '-', color='black',
                       linewidth=1.5, alpha=0.7, zorder=2)
            except (RuntimeError, ValueError):
                pass  # Curve fit failed, skip fitting line
        
        # Shorten parameter name for title: remove common prefixes, use abbreviations
        short_name = param_name.replace('local_unaddressed_', 'lu_').replace('cz_unpaired_gate_', 'cz_up_').replace('cz_paired_gate_', 'cz_pr_').replace('global_', 'g_').replace('local_', 'l_')
        ax.set_title(short_name, fontsize=7, fontweight='bold')
        ax.set_ylabel('Fidelity', fontsize=7)
        ax.set_ylim([0, 1.05])
        ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
        ax.tick_params(labelsize=6)
        
        if np.max(param_values) < 0.01:
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    
    plt.suptitle(f'Noise Analysis: {get_model_name()}', fontsize=12, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution routine for comprehensive noise analysis."""
    global _USE_TWO_ZONE
    
    # Prompt user for model selection
    print("\n" + "="*80)
    print("NOISE MODEL SELECTION")
    print("="*80)
    print("  1. GeminiOneZoneNoiseModel (15 parameters)")
    print("  2. GeminiTwoZoneNoiseModel (21 parameters - includes sitter/mover)")
    
    while True:
        choice = input("\nSelect noise model [1/2]: ").strip()
        if choice == '1':
            _USE_TWO_ZONE = False
            break
        elif choice == '2':
            _USE_TWO_ZONE = True
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")
    
    noise_params = get_noise_params()
    model_name = get_model_name()
    
    print("\n" + "="*80)
    print(f"COMPREHENSIVE NOISE TESTING: {model_name}")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    SCALING_COEFFS = np.linspace(0.1, 3.0, 30)
    PARAM_ITERATIONS = 20  # Per parameter
    SHOTS_PER_TEST = 500
    
    print(f"\nConfiguration:")
    print(f"  - Model: {model_name}")
    print(f"  - Scaling coefficients: {len(SCALING_COEFFS)} points")
    print(f"  - Parameters to sweep: {len(noise_params)}")
    print(f"  - Iterations per parameter: {PARAM_ITERATIONS}")
    print(f"  - Shots per test: {SHOTS_PER_TEST}")
    print(f"  - Parallel workers: {NUM_WORKERS}")
    print(f"  - Total simulations: {len(SCALING_COEFFS) + len(noise_params) * PARAM_ITERATIONS}")
    
    # ---- ANALYSIS 1: Scaling Coefficient ----
    print("\n" + "-"*80)
    print("ANALYSIS 1: Scaling Coefficient Study")
    print("-"*80)
    
    coeffs, scaling_fidelities = analyze_scaling_coefficients(
        coeff_range=SCALING_COEFFS, 
        shots=SHOTS_PER_TEST, 
        verbose=True
    )
    
    print(f"\nScaling Summary: min={scaling_fidelities.min():.4f}, max={scaling_fidelities.max():.4f}, mean={scaling_fidelities.mean():.4f}")
    
    # ---- ANALYSIS 2: All Custom Parameters (BATCHED) ----
    print("\n" + "-"*80)
    print(f"ANALYSIS 2: All {len(noise_params)} Parameter Sweeps (Batched Parallel Execution)")
    print("-"*80)
    
    param_results_dict = analyze_all_parameters_batched(
        iterations=PARAM_ITERATIONS,
        shots=SHOTS_PER_TEST,
        verbose=True
    )
    
    # Convert to list format and compute summaries
    all_param_results = []
    param_summaries = {}
    
    print("\n  Per-parameter summaries:")
    for param_name in noise_params:
        param_values, fidelities = param_results_dict[param_name]
        all_param_results.append((param_name, param_values, fidelities))
        
        sensitivity = (fidelities[0] - fidelities[-1]) / param_values[-1] if param_values[-1] > 0 else 0
        drop = fidelities[0] - fidelities[-1]
        
        param_summaries[param_name] = {
            'min_fidelity': float(fidelities.min()),
            'max_fidelity': float(fidelities.max()),
            'fidelity_drop': float(drop),
            'sensitivity': float(sensitivity),
        }
        print(f"    {param_name:30s} drop={drop:+.4f}")
    
    # ---- RANK PARAMETERS BY IMPACT ----
    print("\n" + "-"*80)
    print("PARAMETER IMPACT RANKING (by fidelity drop)")
    print("-"*80)
    
    ranked = sorted(param_summaries.items(), key=lambda x: x[1]['fidelity_drop'], reverse=True)
    for rank, (name, stats) in enumerate(ranked, 1):
        print(f"  {rank:2d}. {name:30s} drop={stats['fidelity_drop']:.4f}")
    
    # ---- VISUALIZATION ----
    print("\n" + "-"*80)
    print("GENERATING VISUALIZATIONS")
    print("-"*80)
    
    fig = plot_individual_analyses(coeffs, scaling_fidelities, all_param_results)
    
    # Use model-specific filename
    png_filename = 'noise_analysis_two_zone.png' if _USE_TWO_ZONE else 'noise_analysis_one_zone.png'
    fig.savefig(png_filename, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {png_filename}")
    
    # ---- SAVE RESULTS ----
    print("\n" + "-"*80)
    print("SAVING RESULTS")
    print("-"*80)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'model': model_name,
        'configuration': {
            'scaling_coefficients': len(SCALING_COEFFS),
            'param_iterations': PARAM_ITERATIONS,
            'shots_per_test': SHOTS_PER_TEST,
            'total_simulations': len(SCALING_COEFFS) + len(noise_params) * PARAM_ITERATIONS,
        },
        'scaling_analysis': {
            'coefficients': coeffs.tolist(),
            'fidelities': scaling_fidelities.tolist(),
        },
        'parameter_analyses': {
            name: {
                'values': vals.tolist(),
                'fidelities': fids.tolist(),
                **param_summaries[name]
            }
            for name, vals, fids in all_param_results
        },
        'impact_ranking': [name for name, _ in ranked],
    }
    
    # Use model-specific JSON filename
    json_filename = 'noise_analysis_two_zone_results.json' if _USE_TWO_ZONE else 'noise_analysis_one_zone_results.json'
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved: {json_filename}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\n📊 Model: {model_name}")
    print(f"📊 Analyzed {len(noise_params)} noise parameters + scaling coefficient")
    print(f"📈 Most impactful: {ranked[0][0]} (drop={ranked[0][1]['fidelity_drop']:.4f})")
    print(f"📉 Least impactful: {ranked[-1][0]} (drop={ranked[-1][1]['fidelity_drop']:.4f})")
    
    plt.show()
    
    return results


if __name__ == "__main__":
    results = main()
