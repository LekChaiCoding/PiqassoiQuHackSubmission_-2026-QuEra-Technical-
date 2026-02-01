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

# Optimize for parallel execution
NUM_WORKERS = min(os.cpu_count() or 4, 10)


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
    injection_t_only(q)
    
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
    injection_h_then_t(q)
    
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
    cirq_circuit = noise.transform_circuit(_get_circuit(variant), model=noise_model)
    stim_circuit = bloqade.stim.Circuit(load_circuit(cirq_circuit))
    samples = np.array(stim_circuit.compile_sampler().sample(shots=shots))
    return find_good_rate(samples)


# ============================================================================
# NOISE PARAMETERS TO TEST
# ============================================================================

_NOISE_PARAMS = [
    'cz_unpaired_gate_px',
    'cz_unpaired_gate_py',
    'cz_unpaired_gate_pz',
]

_ZERO_PARAMS = {p: 0.0 for p in _NOISE_PARAMS}


# ============================================================================
# PARALLEL TEST WORKERS
# ============================================================================

def _run_scaling_test(args):
    """Worker for scaling coefficient test."""
    variant, coeff, shots = args
    noise_model = noise.GeminiOneZoneNoiseModel(scaling_factor=coeff)
    fidelity = run_simulation(variant, noise_model, shots=shots)
    return (variant, coeff), fidelity


def _run_param_test(args):
    """Worker for parameter sweep test."""
    variant, param_name, param_val, shots = args
    params = _ZERO_PARAMS.copy()
    params[param_name] = param_val
    noise_model = noise.GeminiOneZoneNoiseModel(**params)
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
    
    for param_name in _NOISE_PARAMS:
        param_values = np.linspace(0, 5e-2, iterations)
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
    for param_name in _NOISE_PARAMS:
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


def _power_law(x, A, n):
    return A * np.power(x, -n)


def _logarithmic(x, A, b):
    return A - b * np.log(x)


def _r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0


def plot_comparison(coeffs, scaling_results, param_results):
    """Create comparison plots for both magic state prep variants."""
    
    fig = plt.figure(figsize=(16, 10))
    
    # Color scheme
    colors = {
        't_only': '#e74c3c',      # Red
        'h_then_t': '#3498db'     # Blue
    }
    labels = {
        't_only': 'Non-Magic',
        'h_then_t': 'Magic'
    }
    
    # ---- Top Plot: Scaling Coefficient Comparison ----
    ax_top = fig.add_subplot(2, 2, (1, 2))
    
    for variant in ['t_only', 'h_then_t']:
        fids = scaling_results[variant]
        ax_top.scatter(coeffs, fids, s=60, alpha=0.7, color=colors[variant],
                      edgecolors='black', linewidth=0.5, label=f'{labels[variant]} (data)')
        
        # Fit exponential decay
        try:
            popt, _ = curve_fit(_exp_decay, coeffs, fids, p0=[fids[0], 0.5], maxfev=10000)
            fit_x = np.linspace(coeffs.min(), coeffs.max(), 100)
            r2 = _r_squared(fids, _exp_decay(coeffs, *popt))
            ax_top.plot(fit_x, _exp_decay(fit_x, *popt), '--', color=colors[variant],
                       linewidth=2, alpha=0.8, 
                       label=f'{labels[variant]}: {popt[0]:.3f}·e^(-{popt[1]:.3f}c) [R²={r2:.3f}]')
        except (RuntimeError, ValueError):
            pass
    
    ax_top.set_xlabel('Scaling Coefficient', fontsize=11, fontweight='bold')
    ax_top.set_ylabel('Fidelity', fontsize=11, fontweight='bold')
    ax_top.set_title('Scaling Coefficient: Non-Magic vs Magic State Prep', fontsize=12, fontweight='bold')
    ax_top.set_ylim([0, 1.05])
    ax_top.grid(True, linestyle='--', alpha=0.4)
    ax_top.legend(fontsize=9, loc='best')
    
    # ---- Bottom Plots: Parameter Sweeps ----
    param_names = list(param_results.keys())
    short_names = {
        'cz_unpaired_gate_px': 'cz_up_px',
        'cz_unpaired_gate_py': 'cz_up_py',
        'cz_unpaired_gate_pz': 'cz_up_pz'
    }
    
    for i, param_name in enumerate(param_names):
        ax = fig.add_subplot(2, 3, 4 + i)
        data = param_results[param_name]
        pvals = data['values']
        
        for variant in ['t_only', 'h_then_t']:
            fids = data[variant]
            ax.scatter(pvals, fids, s=40, alpha=0.7, color=colors[variant],
                      edgecolors='black', linewidth=0.5, label=labels[variant])
            
            # Fit exponential decay
            if np.std(fids) > 0.01:
                try:
                    popt, _ = curve_fit(_exp_decay, pvals, fids, p0=[fids[0], 50], maxfev=10000)
                    fit_x = np.linspace(pvals.min(), pvals.max(), 100)
                    ax.plot(fit_x, _exp_decay(fit_x, *popt), '--', color=colors[variant],
                           linewidth=1.5, alpha=0.7)
                except (RuntimeError, ValueError):
                    pass
        
        ax.set_xlabel(short_names.get(param_name, param_name), fontsize=10, fontweight='bold')
        ax.set_ylabel('Fidelity', fontsize=10)
        ax.set_title(short_names.get(param_name, param_name), fontsize=10, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(fontsize=8)
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    
    plt.suptitle('Magic State Prep Comparison: Non-Magic vs Magic', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution routine for magic state prep comparison."""
    
    print("\n" + "="*80)
    print("MAGIC STATE PREPARATION COMPARISON")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nComparing:")
    print("  • Non-Magic:  magicstateprep applies T gate only")
    print("  • Magic:      magicstateprep applies H then T gate")
    
    # Configuration
    SCALING_COEFFS = np.linspace(0.1, 3.0, 25)
    PARAM_ITERATIONS = 15
    SHOTS_PER_TEST = 500
    
    total_tests = 2 * len(SCALING_COEFFS) + 2 * len(_NOISE_PARAMS) * PARAM_ITERATIONS
    
    print(f"\nConfiguration:")
    print(f"  - Scaling coefficients: {len(SCALING_COEFFS)} points × 2 variants")
    print(f"  - Parameters to test: {_NOISE_PARAMS}")
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
    for param_name in _NOISE_PARAMS:
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
    fig.savefig('magic_state_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: magic_state_comparison.png")
    
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
            'parameters_tested': _NOISE_PARAMS,
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
    
    with open('magic_state_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("✓ Saved: magic_state_comparison_results.json")
    
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
