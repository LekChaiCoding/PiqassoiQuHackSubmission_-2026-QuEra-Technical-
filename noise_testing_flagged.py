"""
noise_testing_flagged.py - Parallel noise analysis: A3 with flags vs without flags
==================================================================================

Runs scaling and parameter sweeps for MULTIPLE noise model types, for BOTH the
standard A3 circuit (no flags) and the A3 circuit with flags. Each worker runs
both circuits for a single noise configuration.

NOISE MODELS (and why we chose them):
-------------------------------------
1. GeminiOneZoneNoiseModel (scaling_factor)
   - Single-zone QuEra Gemini-style noise: one set of Pauli error rates (global,
     local, gate, etc.) scaled by a single factor. Well-parameterized (15 params).
   - Chosen: baseline hardware-like model used in noisetesting.py; good for
     comparing flagged vs unflagged under realistic device noise.

2. Symmetric global Pauli (depolarizing-style)
   - GeminiOneZoneNoiseModel with only global_px = global_py = global_pz = p
     (all other params 0). Uniform Pauli errors on every qubit after every gate.
   - Chosen: simple theoretical benchmark; depolarizing noise is standard in
     QEC literature, so we compare how flagged vs unflagged behave under
     symmetric, unstructured noise vs the structured Gemini model.

Parallelization:
  - Simulator: Tsim for circuit sampling.
  - CPU only: ProcessPoolExecutor with NUM_WORKERS (default: all cores; set NOISE_TEST_WORKERS).
  - Optional threads: set NOISE_TEST_USE_THREADS=1 for ThreadPoolExecutor.
  - Fidelity is computed on CPU (vectorized NumPy); no GPU dependency.

Usage:
  python noise_testing_flagged.py
  NOISE_TEST_WORKERS=16 python noise_testing_flagged.py   # use 16 processes

Outputs (plots):
  - noise_analysis_flagged_vs_unflagged.png  (scaling + param grid, all models)
  - noise_analysis_flagged_drops.png         (per-parameter fidelity drop)
  - noise_analysis_decay_rates.png           (exponential decay rate per model)
  - noise_analysis_error_vs_scaling.png     (error rate 1−fidelity vs scaling)
  - noise_analysis_param_heatmap.png        (15 params × value index heatmap)
  - noise_analysis_summary.png               (2×2 summary dashboard)
  - noise_analysis_flagged_results.json
"""

from bloqade import squin
from bloqade.types import Qubit
from kirin.dialects import ilist
from bloqade.cirq_utils import load_circuit, emit_circuit, noise
import bloqade.tsim
import numpy as np

from math import pi
import matplotlib.pyplot as plt
from typing import Literal
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import os
from scipy.optimize import curve_fit

# Parallel CPU: workers (NOISE_TEST_WORKERS), optional threads (NOISE_TEST_USE_THREADS=1).
_num_cpus = os.cpu_count() or 4
NUM_WORKERS = int(os.environ.get("NOISE_TEST_WORKERS", _num_cpus))
NUM_WORKERS = max(1, min(NUM_WORKERS, 64))  # cap 64 to avoid process thrashing


# ============================================================================
# CIRCUIT DEFINITIONS (from noisetesting.py + flagged from PiqassoPartII)
# ============================================================================

@squin.kernel
def magicstateprep(qubits, ind):
    squin.t(qubits[ind])


@squin.kernel
def injection(q: ilist.IList[Qubit, Literal[7]]):
    squin.reset(q[0:2])
    squin.reset(q[3:7])
    magicstateprep(q, 2)
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
    steane_encode_zero_on(q)
    for i in range(7):
        squin.h(q[i])


@squin.kernel
def steane_encode_zero_on_with_flags(q: ilist.IList[Qubit, Literal[7]], flag: Qubit):
    """Latch before encode: CX from 3rd, 5th, 6th ancilla to flag, then encode."""
    squin.cx(q[2], flag)
    squin.cx(q[4], flag)
    squin.cx(q[5], flag)
    steane_encode_zero_on(q)


@squin.kernel
def steane_encode_plus_on_with_flags(q: ilist.IList[Qubit, Literal[7]], flag: Qubit):
    """Latch before encode: CX from 3rd, 5th, 6th ancilla to flag, then encode."""
    squin.cx(q[2], flag)
    squin.cx(q[4], flag)
    squin.cx(q[5], flag)
    steane_encode_plus_on(q)


@squin.kernel
def a3_circuit():
    q = squin.qalloc(21)
    injection(q)
    steane_encode_plus_on(q[7:14])
    for i in range(7):
        squin.cx(q[i], q[i + 7])
    steane_encode_zero_on(q[14:21])
    for i in range(7):
        squin.cx(q[i + 14], q[i])
    for i in range(7):
        squin.h(q[i + 14])
    for i in range(7, 21):
        squin.measure(q[i])


@squin.kernel
def a3_circuit_with_flags():
    q = squin.qalloc(23)
    injection(q[0:7])
    steane_encode_plus_on_with_flags(q[7:14], q[21])
    for i in range(7):
        squin.cx(q[i], q[i + 7])
    steane_encode_zero_on_with_flags(q[14:21], q[22])
    for i in range(7):
        squin.cx(q[i + 14], q[i])
    for i in range(7):
        squin.h(q[i + 14])
    for i in range(7, 23):
        squin.measure(q[i])


# ============================================================================
# FIDELITY
# ============================================================================

SYND_INDICES = [[1, 3, 5, 7], [4, 5, 6, 7], [2, 3, 6, 7]]
# 0-based for vectorized indexing
SYND_INDICES_0 = [[i - 1 for i in idx] for idx in SYND_INDICES]


def _check_syndrome(bits, indices):
    parity = np.prod(1 - 2 * np.asarray(bits)[np.array(indices) - 1])
    return parity == 1


def _syndrome_parities_vectorized(bits_2d, indices_0):
    """bits_2d: (n_samples, 7). indices_0: list of int. Returns (n_samples,) parity."""
    factors = 1 - 2 * bits_2d[:, indices_0]
    return np.prod(factors, axis=1)


def find_good_rate(samples):
    """Vectorized CPU: compute good (all 6 syndromes satisfied) rate over all samples."""
    samples = np.asarray(samples, dtype=np.int8)
    n = samples.shape[0]
    if n == 0:
        return 0.0
    x_bits = samples[:, :7]
    z_bits = samples[:, 7:14]
    ok = np.ones(n, dtype=bool)
    for idx0 in SYND_INDICES_0:
        ok &= _syndrome_parities_vectorized(x_bits, idx0) == 1
        ok &= _syndrome_parities_vectorized(z_bits, idx0) == 1
    return np.sum(ok) / n


# ============================================================================
# CIRCUIT CACHES AND SIMULATION
# ============================================================================

_cached_circuit = None
_cached_circuit_flags = None


def _get_circuit():
    global _cached_circuit
    if _cached_circuit is None:
        _cached_circuit = emit_circuit(a3_circuit)
    return _cached_circuit


def _get_circuit_flags():
    global _cached_circuit_flags
    if _cached_circuit_flags is None:
        _cached_circuit_flags = emit_circuit(a3_circuit_with_flags)
    return _cached_circuit_flags


def run_simulation_no_flags(noise_model, shots=500):
    cirq_circuit = noise.transform_circuit(_get_circuit(), model=noise_model)
    squin_circ = load_circuit(cirq_circuit)
    tsim_circuit = bloqade.tsim.Circuit(squin_circ)
    samples = np.array(tsim_circuit.compile_sampler().sample(shots=shots))
    return find_good_rate(samples)


def run_simulation_with_flags(noise_model, shots=500):
    cirq_circuit = noise.transform_circuit(_get_circuit_flags(), model=noise_model)
    squin_circ = load_circuit(cirq_circuit)
    tsim_circuit = bloqade.tsim.Circuit(squin_circ)
    samples = np.array(tsim_circuit.compile_sampler().sample(shots=shots))
    flag_x = samples[:, 14]
    flag_z = samples[:, 15]
    kept = (flag_x == 0) & (flag_z == 0)
    n_kept = np.sum(kept)
    if n_kept == 0:
        return 0.0
    return find_good_rate(samples[kept][:, 0:14])


# ============================================================================
# NOISE MODEL TYPES (build a model from a scaling coefficient)
# ============================================================================

NOISE_PARAMS = [
    'global_px', 'global_py', 'global_pz',
    'local_px', 'local_py', 'local_pz',
    'local_unaddressed_px', 'local_unaddressed_py', 'local_unaddressed_pz',
    'cz_paired_gate_px', 'cz_paired_gate_py', 'cz_paired_gate_pz',
    'cz_unpaired_gate_px', 'cz_unpaired_gate_py', 'cz_unpaired_gate_pz',
]
ZERO_PARAMS = {p: 0.0 for p in NOISE_PARAMS}

# Build a noise model from scaling coefficient c (used in scaling sweep).
# Symmetric Pauli: global_px=py=pz = c * 1e-3 so c in [0.1, 3] gives p in [1e-4, 3e-3].
def _build_gemini_one(c):
    return noise.GeminiOneZoneNoiseModel(scaling_factor=float(c))

def _build_symmetric_pauli(c):
    p = float(c) * 1e-3
    return noise.GeminiOneZoneNoiseModel(
        global_px=p, global_py=p, global_pz=p,
        local_px=0, local_py=0, local_pz=0,
        local_unaddressed_px=0, local_unaddressed_py=0, local_unaddressed_pz=0,
        cz_paired_gate_px=0, cz_paired_gate_py=0, cz_paired_gate_pz=0,
        cz_unpaired_gate_px=0, cz_unpaired_gate_py=0, cz_unpaired_gate_pz=0,
    )

NOISE_MODEL_SCALING_BUILDERS = {
    'GeminiOneZone': _build_gemini_one,
    'SymmetricPauli': _build_symmetric_pauli,
}

# ============================================================================
# PARALLEL WORKERS (each runs BOTH circuits for one noise config)
# ============================================================================


def _run_scaling_both(args):
    """Run both no-flags and with-flags for one (model_key, coeff). args = (model_key, coeff, shots)."""
    model_key, coeff, shots = args
    build_fn = NOISE_MODEL_SCALING_BUILDERS[model_key]
    nm = build_fn(coeff)
    f_no = run_simulation_no_flags(nm, shots=shots)
    f_fl = run_simulation_with_flags(nm, shots=shots)
    return model_key, coeff, f_no, f_fl


def _run_param_both(args):
    """Run both no-flags and with-flags for one (param_name, param_val). args = (param_name, param_val, shots)."""
    param_name, param_val, shots = args
    params = ZERO_PARAMS.copy()
    params[param_name] = float(param_val)
    nm = noise.GeminiOneZoneNoiseModel(**params)
    f_no = run_simulation_no_flags(nm, shots=shots)
    f_fl = run_simulation_with_flags(nm, shots=shots)
    return (param_name, param_val), f_no, f_fl


def _get_executor():
    """Use ThreadPoolExecutor if requested (Stim releases GIL); else ProcessPoolExecutor."""
    if os.environ.get("NOISE_TEST_USE_THREADS", "").strip() == "1":
        return ThreadPoolExecutor(max_workers=NUM_WORKERS)
    return ProcessPoolExecutor(max_workers=NUM_WORKERS)


def run_scaling_parallel(coeff_range, shots=500, model_keys=None, verbose=True):
    """Parallel scaling for multiple noise model types. Each worker runs both circuits for one (model_key, coeff)."""
    if model_keys is None:
        model_keys = list(NOISE_MODEL_SCALING_BUILDERS.keys())
    args_list = [(mk, c, shots) for mk in model_keys for c in coeff_range]
    total = len(args_list)
    results_raw = []
    with _get_executor() as executor:
        futures = {executor.submit(_run_scaling_both, a): a for a in args_list}
        for i, future in enumerate(as_completed(futures), 1):
            model_key, coeff, f_no, f_fl = future.result()
            results_raw.append((model_key, coeff, f_no, f_fl))
            if verbose and (i % 15 == 0 or i == total):
                print(f"  Scaling [{i}/{total}] {model_key} c={coeff:.2f}  no_flags={f_no:.4f}  with_flags={f_fl:.4f}")
    # Aggregate by model_key, sort by coeff
    out = {}
    for mk in model_keys:
        subset = [(c, f_no, f_fl) for (m, c, f_no, f_fl) in results_raw if m == mk]
        coeffs_s = np.array([x[0] for x in subset])
        f_no_s = np.array([x[1] for x in subset])
        f_fl_s = np.array([x[2] for x in subset])
        order = np.argsort(coeffs_s)
        out[mk] = (coeffs_s[order], f_no_s[order], f_fl_s[order])
    return out


def run_params_parallel(iterations=20, shots=500, verbose=True):
    """Parallel param sweep: each worker runs both circuits for one (param, value). Returns two dicts."""
    all_args = []
    param_ranges = {}
    for param_name in NOISE_PARAMS:
        prange = (0, 5e-2) if 'gate' in param_name.lower() else (0, 2e-3)
        param_values = np.linspace(prange[0], prange[1], iterations)
        param_ranges[param_name] = param_values
        for pv in param_values:
            all_args.append((param_name, pv, shots))
    total = len(all_args)
    results_no = {}
    results_fl = {}
    with _get_executor() as executor:
        futures = {executor.submit(_run_param_both, a): a for a in all_args}
        for i, future in enumerate(as_completed(futures), 1):
            key, f_no, f_fl = future.result()
            results_no[key] = f_no
            results_fl[key] = f_fl
            if verbose and (i % 80 == 0 or i == total):
                print(f"  Params [{i}/{total}] ({100*i/total:.0f}%)")
    param_results_no = {}
    param_results_fl = {}
    for param_name in NOISE_PARAMS:
        pvals = param_ranges[param_name]
        param_results_no[param_name] = (pvals, np.array([results_no[(param_name, pv)] for pv in pvals]))
        param_results_fl[param_name] = (pvals, np.array([results_fl[(param_name, pv)] for pv in pvals]))
    return param_results_no, param_results_fl


# ============================================================================
# PLOTTING (comparison: flagged vs not flagged)
# ============================================================================

def _exp_decay(x, A, b):
    return A * np.exp(-b * x)


def _r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0


def _short_name(name):
    return name.replace('local_unaddressed_', 'lu_').replace('cz_unpaired_gate_', 'cz_up_').replace('cz_paired_gate_', 'cz_pr_').replace('global_', 'g_').replace('local_', 'l_')


def plot_all_comparisons(scaling_results_by_model, param_results_no, param_results_fl, outpath='noise_analysis_flagged_vs_unflagged.png'):
    """One row per noise model (scaling: no-flags vs with-flags), then diff/mean row, then param grid (GeminiOneZone)."""
    model_keys = list(scaling_results_by_model.keys())
    n_models = len(model_keys)
    n_params = len(NOISE_PARAMS)
    n_cols = 4
    n_rows_param = (n_params + n_cols - 1) // n_cols
    # Rows: n_models scaling + 1 (diff/mean) + n_rows_param (param grid)
    n_total_rows = n_models + 1 + n_rows_param
    fig = plt.figure(figsize=(16, 3 * n_models + 4 + 3 * n_rows_param))
    colors_no = '#2980b9'
    colors_fl = '#e74c3c'
    # ---- One scaling plot per noise model ----
    for row, model_key in enumerate(model_keys):
        coeffs, f_no, f_fl = scaling_results_by_model[model_key]
        ax = fig.add_subplot(n_total_rows, n_cols, row * n_cols + 1)
        ax.scatter(coeffs, f_no, s=50, alpha=0.8, color=colors_no, edgecolors='black', linewidth=1, label='No flags', zorder=5)
        ax.scatter(coeffs, f_fl, s=50, alpha=0.8, color=colors_fl, edgecolors='black', linewidth=1, label='With flags (kept only)', zorder=5)
        try:
            popt = curve_fit(_exp_decay, coeffs, f_no, p0=[f_no[0], 0.5], maxfev=10000)[0]
            fit_x = np.linspace(coeffs.min(), coeffs.max(), 100)
            ax.plot(fit_x, _exp_decay(fit_x, *popt), '-', color=colors_no, linewidth=2, alpha=0.6)
        except (RuntimeError, ValueError):
            pass
        try:
            popt2 = curve_fit(_exp_decay, coeffs, f_fl, p0=[f_fl[0], 0.5], maxfev=10000)[0]
            ax.plot(fit_x, _exp_decay(fit_x, *popt2), '--', color=colors_fl, linewidth=2, alpha=0.6)
        except (RuntimeError, ValueError):
            pass
        ax.set_xlabel('Scaling (c)', fontsize=9)
        ax.set_ylabel('Fidelity', fontsize=9)
        ax.set_title(f'{model_key}: no flags vs with flags', fontsize=10, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, linestyle='--', alpha=0.4)
    # ---- Row n_models: Fidelity diff (all models) and mean bar (all models) ----
    ax_diff = fig.add_subplot(n_total_rows, n_cols, n_models * n_cols + 1)
    for model_key in model_keys:
        coeffs, f_no, f_fl = scaling_results_by_model[model_key]
        diff = f_fl - f_no
        ax_diff.plot(coeffs, diff, 'o-', linewidth=1.5, markersize=4, label=model_key)
    ax_diff.axhline(0, color='gray', linestyle='--')
    ax_diff.set_xlabel('Scaling (c)', fontsize=9)
    ax_diff.set_ylabel('Fidelity diff (with − no flags)', fontsize=9)
    ax_diff.set_title('Fidelity difference vs scaling (all models)', fontsize=10, fontweight='bold')
    ax_diff.legend(loc='best', fontsize=7)
    ax_diff.grid(True, linestyle='--', alpha=0.4)
    ax_mean = fig.add_subplot(n_total_rows, n_cols, n_models * n_cols + 2)
    x_pos = np.arange(n_models * 2)
    means_no = [np.mean(scaling_results_by_model[mk][1]) for mk in model_keys]
    means_fl = [np.mean(scaling_results_by_model[mk][2]) for mk in model_keys]
    width = 0.35
    ax_mean.bar(x_pos[::2] - width/2, means_no, width, label='No flags', color=colors_no, edgecolor='black')
    ax_mean.bar(x_pos[1::2] - width/2, means_fl, width, label='With flags', color=colors_fl, edgecolor='black')
    ax_mean.set_xticks(x_pos[::2])
    ax_mean.set_xticklabels(model_keys, fontsize=8)
    ax_mean.set_ylabel('Mean fidelity', fontsize=9)
    ax_mean.set_title('Mean fidelity (scaling sweep)', fontsize=10, fontweight='bold')
    ax_mean.set_ylim(0, 1.05)
    ax_mean.legend(loc='best', fontsize=7)
    # ---- Param grid (GeminiOneZone only) ----
    base = (n_models + 1) * n_cols
    for i, param_name in enumerate(NOISE_PARAMS):
        ax = fig.add_subplot(n_total_rows, n_cols, base + i + 1)
        pvals, f_no = param_results_no[param_name]
        _, f_fl = param_results_fl[param_name]
        ax.scatter(pvals, f_no, s=30, alpha=0.7, color=colors_no, edgecolors='black', linewidth=0.5, label='No flags')
        ax.scatter(pvals, f_fl, s=30, alpha=0.7, color=colors_fl, edgecolors='black', linewidth=0.5, label='With flags')
        ax.set_title(_short_name(param_name), fontsize=7, fontweight='bold')
        ax.set_ylabel('Fidelity', fontsize=6)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=5, loc='best')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.tick_params(labelsize=5)
        if np.max(pvals) < 0.01:
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    for j in range(len(NOISE_PARAMS), n_rows_param * n_cols):
        ax = fig.add_subplot(n_total_rows, n_cols, base + j + 1)
        ax.set_visible(False)
    plt.suptitle('Noise analysis: flagged vs not flagged (multiple noise models)', fontsize=12, fontweight='bold', y=0.998)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"  Saved: {outpath}")
    return fig


def plot_per_param_drop(param_results_no, param_results_fl, outpath='noise_analysis_flagged_drops.png'):
    """Grouped bar: fidelity drop (start − end) per parameter for no flags vs with flags."""
    drops_no = []
    drops_fl = []
    for param_name in NOISE_PARAMS:
        _, f_no = param_results_no[param_name]
        _, f_fl = param_results_fl[param_name]
        drops_no.append(f_no[0] - f_no[-1])
        drops_fl.append(f_fl[0] - f_fl[-1])
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    x = np.arange(len(NOISE_PARAMS))
    w = 0.35
    ax.bar(x - w/2, drops_no, w, label='No flags', color='#2980b9', edgecolor='black')
    ax.bar(x + w/2, drops_fl, w, label='With flags', color='#e74c3c', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([_short_name(p) for p in NOISE_PARAMS], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Fidelity drop (start − end)', fontsize=10, fontweight='bold')
    ax.set_title('Per-parameter fidelity drop: no flags vs with flags', fontsize=11, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"  Saved: {outpath}")
    return fig


def plot_scaling_decay_rates(scaling_results_by_model, outpath='noise_analysis_decay_rates.png'):
    """Bar chart: fitted exponential decay rate (b in A*exp(-b*x)) per model, no-flags vs with-flags."""
    model_keys = list(scaling_results_by_model.keys())
    decay_no, decay_fl = [], []
    for mk in model_keys:
        coeffs, f_no, f_fl = scaling_results_by_model[mk]
        b_no = b_fl = np.nan
        try:
            (_, b_no), _ = curve_fit(_exp_decay, coeffs, f_no, p0=[f_no[0], 0.5], maxfev=10000)
        except (RuntimeError, ValueError):
            pass
        try:
            (_, b_fl), _ = curve_fit(_exp_decay, coeffs, f_fl, p0=[f_fl[0], 0.5], maxfev=10000)
        except (RuntimeError, ValueError):
            pass
        decay_no.append(b_no if not np.isnan(b_no) else 0)
        decay_fl.append(b_fl if not np.isnan(b_fl) else 0)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    x = np.arange(len(model_keys))
    w = 0.35
    ax.bar(x - w/2, decay_no, w, label='No flags', color='#2980b9', edgecolor='black')
    ax.bar(x + w/2, decay_fl, w, label='With flags', color='#e74c3c', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(model_keys)
    ax.set_ylabel('Decay rate (b)', fontsize=10, fontweight='bold')
    ax.set_title('Exponential decay rate: fidelity ~ A*exp(-b*c)', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"  Saved: {outpath}")
    return fig


def plot_error_vs_scaling(scaling_results_by_model, outpath='noise_analysis_error_vs_scaling.png'):
    """Plot raw error rate (1 - fidelity) vs scaling coefficient for each model."""
    model_keys = list(scaling_results_by_model.keys())
    n = len(model_keys)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for idx, mk in enumerate(model_keys):
        ax = axes[0, idx]
        coeffs, f_no, f_fl = scaling_results_by_model[mk]
        err_no = 1 - f_no
        err_fl = 1 - f_fl
        ax.scatter(coeffs, err_no, s=40, alpha=0.8, color='#2980b9', label='No flags', zorder=5)
        ax.scatter(coeffs, err_fl, s=40, alpha=0.8, color='#e74c3c', label='With flags', zorder=5)
        ax.set_xlabel('Scaling (c)', fontsize=9)
        ax.set_ylabel('Error rate (1 − fidelity)', fontsize=9)
        ax.set_title(mk, fontsize=10, fontweight='bold')
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, linestyle='--', alpha=0.4)
    plt.suptitle('Error rate vs noise scaling', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"  Saved: {outpath}")
    return fig


def plot_param_heatmap(param_results_no, param_results_fl, outpath='noise_analysis_param_heatmap.png'):
    """Heatmap: param index x value index -> fidelity. Two panels: no flags, with flags."""
    n_params = len(NOISE_PARAMS)
    # Build matrix: rows = param index, cols = value index (same for all params = iterations)
    iters = len(param_results_no[NOISE_PARAMS[0]][0])
    mat_no = np.zeros((n_params, iters))
    mat_fl = np.zeros((n_params, iters))
    for i, p in enumerate(NOISE_PARAMS):
        mat_no[i, :] = param_results_no[p][1]
        mat_fl[i, :] = param_results_fl[p][1]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    im1 = ax1.imshow(mat_no, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax1.set_ylabel('Parameter index', fontsize=9)
    ax1.set_title('Fidelity (no flags)', fontsize=10, fontweight='bold')
    ax1.set_yticks(range(n_params))
    ax1.set_yticklabels([_short_name(p) for p in NOISE_PARAMS], fontsize=6)
    plt.colorbar(im1, ax=ax1, label='Fidelity')
    im2 = ax2.imshow(mat_fl, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax2.set_ylabel('Parameter index', fontsize=9)
    ax2.set_xlabel('Parameter value index', fontsize=9)
    ax2.set_title('Fidelity (with flags)', fontsize=10, fontweight='bold')
    ax2.set_yticks(range(n_params))
    ax2.set_yticklabels([_short_name(p) for p in NOISE_PARAMS], fontsize=6)
    plt.colorbar(im2, ax=ax2, label='Fidelity')
    plt.suptitle('Fidelity heatmap: 15 params × value index', fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"  Saved: {outpath}")
    return fig


def plot_summary_dashboard(scaling_results_by_model, param_results_no, param_results_fl, outpath='noise_analysis_summary.png'):
    """2x2 summary: mean fidelity (scaling), mean drop (params), decay rates, error at max coeff."""
    model_keys = list(scaling_results_by_model.keys())
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    colors_no, colors_fl = '#2980b9', '#e74c3c'
    # (0,0): Mean fidelity over scaling sweep per model
    ax = axes[0, 0]
    means_no = [np.mean(scaling_results_by_model[mk][1]) for mk in model_keys]
    means_fl = [np.mean(scaling_results_by_model[mk][2]) for mk in model_keys]
    x = np.arange(len(model_keys))
    ax.bar(x - 0.2, means_no, 0.4, label='No flags', color=colors_no, edgecolor='black')
    ax.bar(x + 0.2, means_fl, 0.4, label='With flags', color=colors_fl, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(model_keys)
    ax.set_ylabel('Mean fidelity')
    ax.set_title('Scaling sweep: mean fidelity', fontweight='bold')
    ax.legend(loc='best', fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    # (0,1): Mean fidelity drop over param sweep (no flags vs with flags)
    ax = axes[0, 1]
    drops_no = [param_results_no[p][1][0] - param_results_no[p][1][-1] for p in NOISE_PARAMS]
    drops_fl = [param_results_fl[p][1][0] - param_results_fl[p][1][-1] for p in NOISE_PARAMS]
    ax.bar(np.arange(len(NOISE_PARAMS)) - 0.2, drops_no, 0.4, label='No flags', color=colors_no, edgecolor='black')
    ax.bar(np.arange(len(NOISE_PARAMS)) + 0.2, drops_fl, 0.4, label='With flags', color=colors_fl, edgecolor='black')
    ax.set_xlabel('Parameter index')
    ax.set_ylabel('Fidelity drop (start − end)')
    ax.set_title('Param sweep: fidelity drop per param', fontweight='bold')
    ax.legend(loc='best', fontsize=7)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    # (1,0): Decay rates
    decay_no, decay_fl = [], []
    for mk in model_keys:
        coeffs, f_no, f_fl = scaling_results_by_model[mk]
        b_no = b_fl = np.nan
        try:
            (_, b_no), _ = curve_fit(_exp_decay, coeffs, f_no, p0=[f_no[0], 0.5], maxfev=10000)
        except (RuntimeError, ValueError):
            pass
        try:
            (_, b_fl), _ = curve_fit(_exp_decay, coeffs, f_fl, p0=[f_fl[0], 0.5], maxfev=10000)
        except (RuntimeError, ValueError):
            pass
        decay_no.append(b_no if not np.isnan(b_no) else 0)
        decay_fl.append(b_fl if not np.isnan(b_fl) else 0)
    ax = axes[1, 0]
    x = np.arange(len(model_keys))
    ax.bar(x - 0.2, decay_no, 0.4, label='No flags', color=colors_no, edgecolor='black')
    ax.bar(x + 0.2, decay_fl, 0.4, label='With flags', color=colors_fl, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(model_keys)
    ax.set_ylabel('Decay rate (b)')
    ax.set_title('Scaling: exp decay rate', fontweight='bold')
    ax.legend(loc='best', fontsize=7)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    # (1,1): Error rate at max scaling coeff
    ax = axes[1, 1]
    err_no, err_fl = [], []
    for mk in model_keys:
        coeffs, f_no, f_fl = scaling_results_by_model[mk]
        err_no.append(1 - f_no[-1])
        err_fl.append(1 - f_fl[-1])
    x = np.arange(len(model_keys))
    ax.bar(x - 0.2, err_no, 0.4, label='No flags', color=colors_no, edgecolor='black')
    ax.bar(x + 0.2, err_fl, 0.4, label='With flags', color=colors_fl, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(model_keys)
    ax.set_ylabel('Error rate (1 − fidelity)')
    ax.set_title('Error at max scaling coeff', fontweight='bold')
    ax.legend(loc='best', fontsize=7)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.suptitle('Noise analysis summary', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"  Saved: {outpath}")
    return fig


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*80)
    print("NOISE TESTING: FLAGGED vs NOT FLAGGED (parallel)")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    SCALING_COEFFS = np.linspace(0.1, 3.0, 25)
    PARAM_ITERATIONS = 18
    SHOTS_PER_TEST = 500
    use_threads = os.environ.get("NOISE_TEST_USE_THREADS", "").strip() == "1"
    print(f"\nConfiguration:")
    print(f"  Simulator: Tsim (bloqade.tsim)  |  Shots: {SHOTS_PER_TEST}")
    print(f"  Scaling points: {len(SCALING_COEFFS)}  |  Param iterations: {PARAM_ITERATIONS}")
    print(f"  Parallel: {NUM_WORKERS} workers ({'threads' if use_threads else 'processes'})")
    print(f"  Total jobs: {len(SCALING_COEFFS)} (scaling) + {len(NOISE_PARAMS) * PARAM_ITERATIONS} (params) = {len(SCALING_COEFFS) + len(NOISE_PARAMS) * PARAM_ITERATIONS}")

    print("\n" + "-"*80)
    print("SCALING COEFFICIENT (parallel: both circuits per coeff, all noise model types)")
    print("-"*80)
    scaling_results_by_model = run_scaling_parallel(SCALING_COEFFS, shots=SHOTS_PER_TEST, verbose=True)
    for model_key, (coeffs_arr, f_no, f_fl) in scaling_results_by_model.items():
        print(f"  {model_key}: no_flags mean={f_no.mean():.4f}  with_flags mean={f_fl.mean():.4f}")

    print("\n" + "-"*80)
    print("ALL 15 PARAMETERS (parallel: both circuits per (param, value)) [GeminiOneZone]")
    print("-"*80)
    param_results_no, param_results_fl = run_params_parallel(iterations=PARAM_ITERATIONS, shots=SHOTS_PER_TEST, verbose=True)

    print("\n" + "-"*80)
    print("PLOTTING")
    print("-"*80)
    plot_all_comparisons(scaling_results_by_model, param_results_no, param_results_fl)
    plot_per_param_drop(param_results_no, param_results_fl)
    plot_scaling_decay_rates(scaling_results_by_model)
    plot_error_vs_scaling(scaling_results_by_model)
    plot_param_heatmap(param_results_no, param_results_fl)
    plot_summary_dashboard(scaling_results_by_model, param_results_no, param_results_fl)

    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {'scaling_points': len(SCALING_COEFFS), 'param_iterations': PARAM_ITERATIONS, 'shots': SHOTS_PER_TEST, 'noise_models': list(NOISE_MODEL_SCALING_BUILDERS.keys())},
        'scaling': {mk: {'coeffs': c.tolist(), 'fidelities_no_flags': f_no.tolist(), 'fidelities_with_flags': f_fl.tolist()} for mk, (c, f_no, f_fl) in scaling_results_by_model.items()},
        'params_no_flags': {p: {'values': param_results_no[p][0].tolist(), 'fidelities': param_results_no[p][1].tolist()} for p in NOISE_PARAMS},
        'params_with_flags': {p: {'values': param_results_fl[p][0].tolist(), 'fidelities': param_results_fl[p][1].tolist()} for p in NOISE_PARAMS},
    }
    with open('noise_analysis_flagged_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("  Saved: noise_analysis_flagged_results.json")

    print("\n" + "="*80)
    print("DONE")
    print("="*80)
    plt.show()
    return results


if __name__ == "__main__":
    main()
