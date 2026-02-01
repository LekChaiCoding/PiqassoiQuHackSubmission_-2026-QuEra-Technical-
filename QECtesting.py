"""
QECtesting.py - Comprehensive Noise Analysis for A3 Circuit
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
from bloqade.cirq_utils import load_circuit, emit_circuit, noise
import bloqade.stim
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from typing import Literal
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from scipy.optimize import curve_fit

# Optimize for Snapdragon X Elite - use all available cores
NUM_WORKERS = min(os.cpu_count() or 4, 10)  # Cap at 10 to avoid overhead


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
    injection(q)
    # Qubits 0-6: logical data
    # Qubits 8-14 + 15-21: ancilla block

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
    parity = 1
    for i in indices:
        parity *= (1 - 2 * bits[i - 1])
    return parity == 1


def find_good_rate(samples):
    """Calculate fidelity: fraction of samples with all 6 syndromes triggered."""
    good = 0
    for sample in samples:
        x_bits, z_bits = sample[0:7], sample[7:14]
        count = sum(_check_syndrome(x_bits, idx) for idx in SYND_INDICES)
        count += sum(_check_syndrome(z_bits, idx) for idx in SYND_INDICES)
        if count == 6:
            good += 1
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
    noise_model = noise.GeminiOneZoneNoiseModel(scaling_factor=coeff)
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

# All noise parameters that GeminiOneZoneNoiseModel accepts
_NOISE_PARAMS = [
    'global_px', 'global_py', 'global_pz',
    'local_px', 'local_py', 'local_pz',
    'local_unaddressed_px', 'local_unaddressed_py', 'local_unaddressed_pz',
    'cz_paired_gate_px', 'cz_paired_gate_py', 'cz_paired_gate_pz',
    'cz_unpaired_gate_px', 'cz_unpaired_gate_py', 'cz_unpaired_gate_pz', 'sitter_px', 
    'sitter_py', 'sitter_pz', 'mover_px', 'mover_py', 'mover_pz',
]


# Pre-built zero params dict (avoid rebuilding every call)
_ZERO_PARAMS = {p: 0.0 for p in _NOISE_PARAMS}

def _run_custom_param_test(args):
    """Worker function for parallel custom parameter tests."""
    param_name, param_val, shots = args
    params = _ZERO_PARAMS.copy()
    params[param_name] = param_val
    noise_model = noise.GeminiOneZoneNoiseModel(**params)
    return (param_name, param_val), run_simulation_with_noise(noise_model, shots=shots)


def analyze_all_parameters_batched(iterations=20, shots=500, verbose=True):
    """
    Run ALL parameter sweeps in a SINGLE parallel pool (much faster).
    Returns dict: {param_name: (param_values, fidelities)}
    """
    # Build all args for all parameters at once
    all_args = []
    param_ranges = {}
    
    for param_name in _NOISE_PARAMS:
        prange = (0, 5e-2) if 'gate' in param_name.lower() else (0, 2e-3)
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
    for param_name in _NOISE_PARAMS:
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
    """Create detailed plots: scaling analysis + grid of all parameter sweeps."""
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
        except:
            pass
    
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
              '#2980b9', '#8e44ad', '#d35400', '#27ae60', '#7f8c8d']
    
    for i, (param_name, param_values, fidelities) in enumerate(all_param_results):
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
            except:
                pass
        
        # Shorten parameter name for title: remove common prefixes, use abbreviations
        short_name = param_name.replace('local_unaddressed_', 'lu_').replace('cz_unpaired_gate_', 'cz_up_').replace('cz_paired_gate_', 'cz_pr_').replace('global_', 'g_').replace('local_', 'l_')
        ax.set_title(short_name, fontsize=7, fontweight='bold')
        ax.set_ylabel('Fidelity', fontsize=7)
        ax.set_ylim([0, 1.05])
        ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
        ax.tick_params(labelsize=6)
        
        if np.max(param_values) < 0.01:
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    
    plt.suptitle('Noise Analysis: GeminiOneZoneNoiseModel', fontsize=12, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution routine for comprehensive noise analysis."""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE NOISE TESTING: ALL PARAMETERS")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    SCALING_COEFFS = np.linspace(0.1, 3.0, 30)
    PARAM_ITERATIONS = 20  # Per parameter (15 params × 20 = 300 additional tests)
    SHOTS_PER_TEST = 500
    
    print(f"\nConfiguration:")
    print(f"  - Scaling coefficients: {len(SCALING_COEFFS)} points")
    print(f"  - Parameters to sweep: {len(_NOISE_PARAMS)}")
    print(f"  - Iterations per parameter: {PARAM_ITERATIONS}")
    print(f"  - Shots per test: {SHOTS_PER_TEST}")
    print(f"  - Parallel workers: {NUM_WORKERS}")
    print(f"  - Total simulations: {len(SCALING_COEFFS) + len(_NOISE_PARAMS) * PARAM_ITERATIONS}")
    
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
    print("ANALYSIS 2: All 15 Parameter Sweeps (Batched Parallel Execution)")
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
    for param_name in _NOISE_PARAMS:
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
    fig.savefig('noise_analysis_all_params.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: noise_analysis_all_params.png")
    
    # ---- SAVE RESULTS ----
    print("\n" + "-"*80)
    print("SAVING RESULTS")
    print("-"*80)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'scaling_coefficients': len(SCALING_COEFFS),
            'param_iterations': PARAM_ITERATIONS,
            'shots_per_test': SHOTS_PER_TEST,
            'total_simulations': len(SCALING_COEFFS) + len(_NOISE_PARAMS) * PARAM_ITERATIONS,
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
    
    with open('noise_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("✓ Saved: noise_analysis_results.json")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\n📊 Analyzed {len(_NOISE_PARAMS)} noise parameters + scaling coefficient")
    print(f"📈 Most impactful: {ranked[0][0]} (drop={ranked[0][1]['fidelity_drop']:.4f})")
    print(f"📉 Least impactful: {ranked[-1][0]} (drop={ranked[-1][1]['fidelity_drop']:.4f})")
    
    plt.show()
    
    return results


if __name__ == "__main__":
    results = main()
