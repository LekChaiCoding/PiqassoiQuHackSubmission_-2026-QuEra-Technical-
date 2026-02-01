"""
QECtesting.py - Multi-Round QEC Analysis for A3 Circuit
========================================================

This module analyzes the effect of multiple QEC rounds on circuit fidelity,
using the Steane [[7,1,3]] code with fault-tolerant syndrome extraction.

Features:
- Multi-round QEC simulation (1-10 rounds)
- Fidelity vs QEC rounds analysis
- QEC rounds × noise scaling 2D analysis
- Threshold behavior visualization
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


# ============================================================================
# SINGLE-ROUND A3 CIRCUIT (baseline)
# ============================================================================

@squin.kernel
def a3_circuit():
    """A3 circuit - single round of Steane syndrome extraction."""
    q = squin.qalloc(21)
    injection(q)
    
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
# MULTI-ROUND QEC CIRCUITS (2-10 rounds)
# ============================================================================
# Each circuit performs N rounds of syndrome extraction with ancilla resets

@squin.kernel
def a3_circuit_2_rounds():
    """A3 circuit with 2 rounds of syndrome extraction."""
    q = squin.qalloc(21)
    injection(q)
    
    # Round 1
    steane_encode_plus_on(q[7:14])
    for i in range(7):
        squin.cx(q[i], q[i+7])
    steane_encode_zero_on(q[14:21])
    for i in range(7):
        squin.cx(q[i+14], q[i])
    for i in range(7):
        squin.h(q[i+14])
    
    # Reset ancillas for round 2
    for i in range(7, 21):
        squin.reset(q[i])
    
    # Round 2
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
def a3_circuit_3_rounds():
    """A3 circuit with 3 rounds of syndrome extraction."""
    q = squin.qalloc(21)
    injection(q)
    
    for _round in range(3):
        if _round > 0:
            for i in range(7, 21):
                squin.reset(q[i])
        
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
def a3_circuit_4_rounds():
    """A3 circuit with 4 rounds of syndrome extraction."""
    q = squin.qalloc(21)
    injection(q)
    
    for _round in range(4):
        if _round > 0:
            for i in range(7, 21):
                squin.reset(q[i])
        
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
def a3_circuit_5_rounds():
    """A3 circuit with 5 rounds of syndrome extraction."""
    q = squin.qalloc(21)
    injection(q)
    
    for _round in range(5):
        if _round > 0:
            for i in range(7, 21):
                squin.reset(q[i])
        
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
def a3_circuit_6_rounds():
    """A3 circuit with 6 rounds of syndrome extraction."""
    q = squin.qalloc(21)
    injection(q)
    
    for _round in range(6):
        if _round > 0:
            for i in range(7, 21):
                squin.reset(q[i])
        
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
def a3_circuit_7_rounds():
    """A3 circuit with 7 rounds of syndrome extraction."""
    q = squin.qalloc(21)
    injection(q)
    
    for _round in range(7):
        if _round > 0:
            for i in range(7, 21):
                squin.reset(q[i])
        
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
def a3_circuit_8_rounds():
    """A3 circuit with 8 rounds of syndrome extraction."""
    q = squin.qalloc(21)
    injection(q)
    
    for _round in range(8):
        if _round > 0:
            for i in range(7, 21):
                squin.reset(q[i])
        
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
def a3_circuit_9_rounds():
    """A3 circuit with 9 rounds of syndrome extraction."""
    q = squin.qalloc(21)
    injection(q)
    
    for _round in range(9):
        if _round > 0:
            for i in range(7, 21):
                squin.reset(q[i])
        
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
def a3_circuit_10_rounds():
    """A3 circuit with 10 rounds of syndrome extraction."""
    q = squin.qalloc(21)
    injection(q)
    
    for _round in range(10):
        if _round > 0:
            for i in range(7, 21):
                squin.reset(q[i])
        
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


# Map round count to circuit function
ROUND_CIRCUITS = {
    1: a3_circuit,
    2: a3_circuit_2_rounds,
    3: a3_circuit_3_rounds,
    4: a3_circuit_4_rounds,
    5: a3_circuit_5_rounds,
    6: a3_circuit_6_rounds,
    7: a3_circuit_7_rounds,
    8: a3_circuit_8_rounds,
    9: a3_circuit_9_rounds,
    10: a3_circuit_10_rounds,
}


# ============================================================================
# FIDELITY AND SYNDROME DETECTION
# ============================================================================

# Syndrome indices for Steane [[7,1,3]] code
SYND_INDICES = [[1, 3, 5, 7], [4, 5, 6, 7], [2, 3, 6, 7]]


def _check_syndrome_vectorized(bits, indices):
    """Vectorized syndrome check: returns True if syndrome triggered."""
    idx_arr = np.array(indices) - 1  # Convert to 0-indexed
    parities = 1 - 2 * bits[:, idx_arr]  # Shape: (n_samples, 4)
    return np.prod(parities, axis=1) == 1


def find_good_rate(samples):
    """Calculate fidelity: fraction of samples with all 6 syndromes triggered (vectorized)."""
    samples = np.atleast_2d(samples)
    x_bits = samples[:, 0:7]
    z_bits = samples[:, 7:14]
    
    syndrome_count = np.zeros(len(samples), dtype=int)
    for idx in SYND_INDICES:
        syndrome_count += _check_syndrome_vectorized(x_bits, idx).astype(int)
        syndrome_count += _check_syndrome_vectorized(z_bits, idx).astype(int)
    
    return np.sum(syndrome_count == 6) / len(samples)


# ============================================================================
# MULTI-ROUND CIRCUIT CACHE
# ============================================================================

# Cache for all circuit round configurations
_circuit_cache = {}


def _get_circuit(rounds=1):
    """Get cached circuit for specified number of QEC rounds."""
    global _circuit_cache
    if rounds not in _circuit_cache:
        if rounds not in ROUND_CIRCUITS:
            raise ValueError(f"No circuit defined for {rounds} rounds. Available: {list(ROUND_CIRCUITS.keys())}")
        _circuit_cache[rounds] = emit_circuit(ROUND_CIRCUITS[rounds])
    return _circuit_cache[rounds]


def run_simulation_with_noise(noise_model, shots=1000, rounds=1):
    """Run QEC circuit with specified rounds and noise model, return fidelity."""
    try:
        circuit = _get_circuit(rounds)
        cirq_circuit = noise.transform_circuit(circuit, model=noise_model)
        stim_circuit = bloqade.stim.Circuit(load_circuit(cirq_circuit))
        samples = np.array(stim_circuit.compile_sampler().sample(shots=shots))
        return find_good_rate(samples)
    except (RuntimeError, ValueError, TypeError) as e:
        print(f"    Warning: Simulation failed for rounds={rounds}: {e}")
        return 0.0


# ============================================================================
# ANALYSIS 1: QEC Rounds vs Fidelity (at fixed noise level)
# ============================================================================

def _run_rounds_test(args):
    """Worker function for parallel QEC rounds tests."""
    rounds, scaling_factor, shots = args
    noise_model = noise.GeminiOneZoneNoiseModel(scaling_factor=scaling_factor)
    fidelity = run_simulation_with_noise(noise_model, shots=shots, rounds=rounds)
    return rounds, fidelity


def analyze_qec_rounds(max_rounds=10, scaling_factor=1.0, shots=500, verbose=True):
    """
    Analyze how fidelity changes with number of QEC rounds at fixed noise level.
    
    Returns:
        rounds_array: np.array of round counts (1 to max_rounds)
        fidelities: np.array of corresponding fidelities
    """
    rounds_range = list(range(1, max_rounds + 1))
    
    if verbose:
        print(f"  Scaling factor: {scaling_factor}")
        print(f"  Running {len(rounds_range)} tests in parallel ({NUM_WORKERS} workers)...")
    
    args_list = [(r, scaling_factor, shots) for r in rounds_range]
    
    results = {}
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(_run_rounds_test, args): args[0] for args in args_list}
        for i, future in enumerate(as_completed(futures), 1):
            rounds, fidelity = future.result()
            results[rounds] = fidelity
            if verbose:
                print(f"  [{i:2d}/{len(rounds_range)}] rounds={rounds} → Fidelity: {fidelity:.4f}")
    
    fidelities = np.array([results[r] for r in rounds_range])
    return np.array(rounds_range), fidelities


# ============================================================================
# ANALYSIS 2: 2D Sweep (QEC Rounds × Noise Scaling)
# ============================================================================

def _run_2d_test(args):
    """Worker function for parallel 2D sweep tests."""
    rounds, scaling_factor, shots = args
    noise_model = noise.GeminiOneZoneNoiseModel(scaling_factor=scaling_factor)
    fidelity = run_simulation_with_noise(noise_model, shots=shots, rounds=rounds)
    return (rounds, round(scaling_factor, 6)), fidelity


def analyze_2d_sweep(max_rounds=10, scaling_range=None, shots=500, verbose=True):
    """
    2D analysis: QEC rounds × noise scaling coefficient.
    
    Returns:
        rounds_array: np.array of round counts
        scaling_array: np.array of scaling factors
        fidelity_matrix: 2D array [rounds, scaling] of fidelities
    """
    if scaling_range is None:
        scaling_range = np.linspace(0.1, 2.0, 15)
    
    rounds_range = list(range(1, max_rounds + 1))
    
    # Build all args (round scaling to avoid float precision issues in dict keys)
    all_args = []
    for r in rounds_range:
        for s in scaling_range:
            all_args.append((r, round(s, 6), shots))
    
    total = len(all_args)
    if verbose:
        print(f"  Grid: {len(rounds_range)} rounds × {len(scaling_range)} scaling factors = {total} tests")
        print(f"  Running in parallel ({NUM_WORKERS} workers)...")
    
    results = {}
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(_run_2d_test, args): args for args in all_args}
        for i, future in enumerate(as_completed(futures), 1):
            key, fidelity = future.result()
            results[key] = fidelity
            if verbose and i % 20 == 0:
                print(f"    Progress: {i}/{total} ({100*i/total:.0f}%)")
    
    if verbose:
        print(f"    Progress: {total}/{total} (100%)")
    
    # Build matrix
    fidelity_matrix = np.zeros((len(rounds_range), len(scaling_range)))
    for i, r in enumerate(rounds_range):
        for j, s in enumerate(scaling_range):
            fidelity_matrix[i, j] = results[(r, round(s, 6))]
    
    return np.array(rounds_range), np.array(scaling_range), fidelity_matrix


# ============================================================================
# ANALYSIS 3: Multiple Noise Levels Comparison
# ============================================================================

def analyze_multiple_noise_levels(max_rounds=10, scaling_factors=None, shots=500, verbose=True):
    """
    Analyze QEC rounds for multiple noise levels (for line plot comparison).
    
    Returns:
        rounds_array: np.array of round counts
        results_dict: {scaling_factor: fidelities_array}
    """
    if scaling_factors is None:
        scaling_factors = [0.2, 0.5, 1.0, 1.5, 2.0]
    
    rounds_range = list(range(1, max_rounds + 1))
    
    # Build all args
    all_args = []
    for scaling in scaling_factors:
        for r in rounds_range:
            all_args.append((r, scaling, shots))
    
    total = len(all_args)
    if verbose:
        print(f"  Noise levels: {scaling_factors}")
        print(f"  Max rounds: {max_rounds}")
        print(f"  Total tests: {total}")
        print(f"  Running in parallel ({NUM_WORKERS} workers)...")
    
    results = {}
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(_run_rounds_test, args): args for args in all_args}
        for i, future in enumerate(as_completed(futures), 1):
            rounds, fidelity = future.result()
            args = futures[future]
            scaling = args[1]
            results[(rounds, scaling)] = fidelity
            if verbose and i % 10 == 0:
                print(f"    Progress: {i}/{total} ({100*i/total:.0f}%)")
    
    if verbose:
        print(f"    Progress: {total}/{total} (100%)")
    
    # Organize results
    results_dict = {}
    for scaling in scaling_factors:
        fidelities = np.array([results[(r, scaling)] for r in rounds_range])
        results_dict[scaling] = fidelities
    
    return np.array(rounds_range), results_dict


# ============================================================================
# VISUALIZATION
# ============================================================================

# Fit functions for curve fitting
def _exp_decay(x, A, b): return A * np.exp(-b * x)
def _r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0


def plot_qec_rounds_analysis(rounds, results_dict, title_suffix=""):
    """
    Plot fidelity vs QEC rounds for multiple noise levels.
    
    Args:
        rounds: np.array of round counts
        results_dict: {scaling_factor: fidelities_array}
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color palette for different noise levels
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c', '#e67e22']
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h']
    
    # Sort scaling factors for consistent legend ordering
    sorted_factors = sorted(results_dict.keys())
    
    for i, scaling in enumerate(sorted_factors):
        fidelities = results_dict[scaling]
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        # Plot data points
        ax.scatter(rounds, fidelities, s=80, color=color, marker=marker,
                   edgecolors='black', linewidth=1, zorder=5,
                   label=f'Noise scaling = {scaling:.1f}')
        
        # Connect with lines
        ax.plot(rounds, fidelities, '-', color=color, linewidth=2, alpha=0.7, zorder=4)
        
        # Fit exponential decay if reasonable variation
        if np.std(fidelities) > 0.01 and len(rounds) > 2:
            try:
                popt, _ = curve_fit(_exp_decay, rounds, fidelities,
                                   p0=[fidelities[0], 0.1], maxfev=5000,
                                   bounds=([0, 0], [1.5, 5]))
                fit_x = np.linspace(rounds.min(), rounds.max(), 100)
                ax.plot(fit_x, _exp_decay(fit_x, *popt), '--', color=color,
                       linewidth=1.5, alpha=0.5, zorder=3)
            except:
                pass
    
    ax.set_xlabel('Number of QEC Rounds', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fidelity', fontsize=12, fontweight='bold')
    ax.set_title(f'Fidelity vs QEC Syndrome Extraction Rounds{title_suffix}', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.set_xlim([0.5, rounds.max() + 0.5])
    ax.set_xticks(rounds)
    ax.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    return fig


def plot_2d_heatmap(rounds, scaling, fidelity_matrix, title_suffix=""):
    """
    Create 2D heatmap of fidelity: QEC rounds vs noise scaling.
    
    Args:
        rounds: np.array of round counts
        scaling: np.array of scaling factors
        fidelity_matrix: 2D array [rounds, scaling] of fidelities
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap
    im = ax.imshow(fidelity_matrix, aspect='auto', origin='lower',
                   extent=[scaling.min(), scaling.max(), rounds.min()-0.5, rounds.max()+0.5],
                   cmap='RdYlGn', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Fidelity', pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    
    # Add contour lines for clarity
    X, Y = np.meshgrid(scaling, rounds)
    contours = ax.contour(X, Y, fidelity_matrix, levels=[0.1, 0.3, 0.5, 0.7, 0.9],
                          colors='black', linewidths=0.8, alpha=0.6)
    ax.clabel(contours, inline=True, fontsize=8, fmt='%.1f')
    
    ax.set_xlabel('Noise Scaling Factor', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of QEC Rounds', fontsize=12, fontweight='bold')
    ax.set_title(f'QEC Performance: Fidelity Heatmap{title_suffix}', fontsize=14, fontweight='bold')
    ax.set_yticks(rounds)
    
    plt.tight_layout()
    return fig


def plot_combined_analysis(rounds, results_dict, scaling, fidelity_matrix):
    """Create combined figure with line plots and heatmap."""
    fig = plt.figure(figsize=(16, 6))
    
    # ---- Left: Line Plot ----
    ax1 = fig.add_subplot(1, 2, 1)
    
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    sorted_factors = sorted(results_dict.keys())
    
    for i, sf in enumerate(sorted_factors):
        fids = results_dict[sf]
        ax1.scatter(rounds, fids, s=60, color=colors[i % len(colors)], 
                    marker=markers[i % len(markers)], edgecolors='black', linewidth=0.8,
                    zorder=5, label=f'Noise = {sf:.1f}')
        ax1.plot(rounds, fids, '-', color=colors[i % len(colors)], linewidth=1.5, alpha=0.7, zorder=4)
    
    ax1.set_xlabel('QEC Rounds', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Fidelity', fontsize=11, fontweight='bold')
    ax1.set_title('Fidelity vs QEC Rounds', fontsize=12, fontweight='bold')
    ax1.set_ylim([0, 1.05])
    ax1.set_xlim([0.5, rounds.max() + 0.5])
    ax1.set_xticks(rounds)
    ax1.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax1.legend(loc='upper right', fontsize=8, framealpha=0.9)
    
    # ---- Right: Heatmap ----
    ax2 = fig.add_subplot(1, 2, 2)
    
    im = ax2.imshow(fidelity_matrix, aspect='auto', origin='lower',
                    extent=[scaling.min(), scaling.max(), rounds.min()-0.5, rounds.max()+0.5],
                    cmap='RdYlGn', vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax2, label='Fidelity', pad=0.02)
    
    X, Y = np.meshgrid(scaling, rounds)
    contours = ax2.contour(X, Y, fidelity_matrix, levels=[0.3, 0.5, 0.7],
                           colors='black', linewidths=0.8, alpha=0.6)
    ax2.clabel(contours, inline=True, fontsize=8, fmt='%.1f')
    
    ax2.set_xlabel('Noise Scaling Factor', fontsize=11, fontweight='bold')
    ax2.set_ylabel('QEC Rounds', fontsize=11, fontweight='bold')
    ax2.set_title('Fidelity Heatmap', fontsize=12, fontweight='bold')
    ax2.set_yticks(rounds)
    
    plt.suptitle('Multi-Round QEC Analysis: Steane [[7,1,3]] Code', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution routine for multi-round QEC analysis."""
    
    print("\n" + "="*80)
    print("MULTI-ROUND QEC ANALYSIS: Steane [[7,1,3]] Code")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    MAX_ROUNDS = 10
    SCALING_FACTORS_LINE = [0.2, 0.5, 0.8, 1.0, 1.5, 2.0]  # For line plot
    SCALING_RANGE_HEATMAP = np.linspace(0.1, 2.5, 20)      # For heatmap
    SHOTS_PER_TEST = 500
    
    print(f"\nConfiguration:")
    print(f"  - Max QEC rounds: {MAX_ROUNDS}")
    print(f"  - Noise levels for line plot: {SCALING_FACTORS_LINE}")
    print(f"  - Heatmap scaling range: {SCALING_RANGE_HEATMAP.min():.1f} to {SCALING_RANGE_HEATMAP.max():.1f} ({len(SCALING_RANGE_HEATMAP)} points)")
    print(f"  - Shots per test: {SHOTS_PER_TEST}")
    print(f"  - Parallel workers: {NUM_WORKERS}")
    print(f"  - Available circuit rounds: {list(ROUND_CIRCUITS.keys())}")
    
    # ---- ANALYSIS 1: Multiple Noise Levels (Line Plot) ----
    print("\n" + "-"*80)
    print("ANALYSIS 1: Fidelity vs QEC Rounds (Multiple Noise Levels)")
    print("-"*80)
    
    rounds, results_dict = analyze_multiple_noise_levels(
        max_rounds=MAX_ROUNDS,
        scaling_factors=SCALING_FACTORS_LINE,
        shots=SHOTS_PER_TEST,
        verbose=True
    )
    
    print("\n  Summary by noise level:")
    for scaling in sorted(results_dict.keys()):
        fids = results_dict[scaling]
        print(f"    Noise={scaling:.1f}: start={fids[0]:.4f}, end={fids[-1]:.4f}, drop={fids[0]-fids[-1]:.4f}")
    
    # ---- ANALYSIS 2: 2D Sweep (Heatmap) ----
    print("\n" + "-"*80)
    print("ANALYSIS 2: 2D Sweep (Rounds × Noise Scaling)")
    print("-"*80)
    
    rounds_2d, scaling_2d, fidelity_matrix = analyze_2d_sweep(
        max_rounds=MAX_ROUNDS,
        scaling_range=SCALING_RANGE_HEATMAP,
        shots=SHOTS_PER_TEST,
        verbose=True
    )
    
    print(f"\n  Fidelity matrix shape: {fidelity_matrix.shape}")
    print(f"  Global min: {fidelity_matrix.min():.4f}, max: {fidelity_matrix.max():.4f}")
    
    # ---- VISUALIZATION ----
    print("\n" + "-"*80)
    print("GENERATING VISUALIZATIONS")
    print("-"*80)
    
    # Line plot
    fig1 = plot_qec_rounds_analysis(rounds, results_dict)
    fig1.savefig('qec_rounds_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: qec_rounds_analysis.png")
    
    # Heatmap
    fig2 = plot_2d_heatmap(rounds_2d, scaling_2d, fidelity_matrix)
    fig2.savefig('qec_heatmap.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: qec_heatmap.png")
    
    # Combined view
    fig3 = plot_combined_analysis(rounds, results_dict, scaling_2d, fidelity_matrix)
    fig3.savefig('qec_combined_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: qec_combined_analysis.png")
    
    # ---- SAVE RESULTS ----
    print("\n" + "-"*80)
    print("SAVING RESULTS")
    print("-"*80)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'max_rounds': MAX_ROUNDS,
            'scaling_factors_line': SCALING_FACTORS_LINE,
            'scaling_range_heatmap': SCALING_RANGE_HEATMAP.tolist(),
            'shots_per_test': SHOTS_PER_TEST,
        },
        'line_plot_data': {
            'rounds': rounds.tolist(),
            'results_by_noise_level': {
                str(k): v.tolist() for k, v in results_dict.items()
            },
        },
        'heatmap_data': {
            'rounds': rounds_2d.tolist(),
            'scaling': scaling_2d.tolist(),
            'fidelity_matrix': fidelity_matrix.tolist(),
        },
        'summary': {
            'global_min_fidelity': float(fidelity_matrix.min()),
            'global_max_fidelity': float(fidelity_matrix.max()),
            'best_config': {
                'rounds': int(rounds_2d[np.unravel_index(fidelity_matrix.argmax(), fidelity_matrix.shape)[0]]),
                'scaling': float(scaling_2d[np.unravel_index(fidelity_matrix.argmax(), fidelity_matrix.shape)[1]]),
            },
        },
    }
    
    with open('qec_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("✓ Saved: qec_analysis_results.json")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\n📊 Analyzed {MAX_ROUNDS} QEC rounds × {len(SCALING_RANGE_HEATMAP)} noise levels")
    print(f"📈 Best fidelity: {fidelity_matrix.max():.4f} at {results['summary']['best_config']}")
    print(f"📉 Worst fidelity: {fidelity_matrix.min():.4f}")
    
    # Show key insight
    print("\n📋 Key Insight:")
    low_noise_drop = results_dict[0.2][0] - results_dict[0.2][-1]
    high_noise_drop = results_dict[2.0][0] - results_dict[2.0][-1]
    print(f"   At low noise (0.2): fidelity drops {low_noise_drop:.4f} over {MAX_ROUNDS} rounds")
    print(f"   At high noise (2.0): fidelity drops {high_noise_drop:.4f} over {MAX_ROUNDS} rounds")
    
    plt.show()
    
    return results


if __name__ == "__main__":
    results = main()
