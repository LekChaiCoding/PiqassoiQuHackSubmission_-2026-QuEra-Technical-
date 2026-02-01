"""
Quick script to plot mover_* and sitter_* noise parameters from two-zone results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Load results
with open('noise_analysis_two_zone_results.json', 'r') as f:
    data = json.load(f)

# Filter for mover and sitter params only
param_analyses = data['parameter_analyses']
target_params = [k for k in param_analyses.keys() if k.startswith('mover_') or k.startswith('sitter_')]

# Exponential decay fit
def exp_decay(x, A, b):
    return A * np.exp(-b * x)

# Create 2x3 subplot grid
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

colors = {
    'sitter_px': '#e74c3c', 'sitter_py': '#c0392b', 'sitter_pz': '#922b21',
    'mover_px': '#3498db', 'mover_py': '#2980b9', 'mover_pz': '#1f618d'
}

for i, param_name in enumerate(target_params):
    ax = axes[i]
    pdata = param_analyses[param_name]
    
    values = np.array(pdata['values'])
    fidelities = np.array(pdata['fidelities'])
    
    color = colors.get(param_name, '#333333')
    
    # Scatter plot
    ax.scatter(values, fidelities, s=50, alpha=0.8, color=color,
               edgecolors='black', linewidth=0.5, label='Data')
    
    # Fit exponential decay
    try:
        popt, _ = curve_fit(exp_decay, values, fidelities, p0=[fidelities[0], 50], maxfev=10000)
        fit_x = np.linspace(values.min(), values.max(), 100)
        ax.plot(fit_x, exp_decay(fit_x, *popt), '--', color=color, linewidth=2,
                label=f'Fit: {popt[0]:.2f}·e^(-{popt[1]:.1f}x)')
    except:
        pass
    
    # Formatting
    ax.set_xlabel('Parameter Value', fontsize=10)
    ax.set_ylabel('Fidelity', fontsize=10)
    ax.set_title(param_name.replace('_', ' ').title(), fontsize=11, fontweight='bold')
    ax.set_ylim([0, 0.7])
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=8, loc='upper right')
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

plt.suptitle('Mover & Sitter Noise Impact (TwoZone Model)', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig('mover_sitter_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved: mover_sitter_analysis.png")
plt.show()
