"""
Quick script to compare scaling coefficient: OneZone vs TwoZone noise models.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Load both results
with open('noise_analysis_results.json', 'r') as f:
    one_zone = json.load(f)

with open('noise_analysis_two_zone_results.json', 'r') as f:
    two_zone = json.load(f)

# Extract scaling data
coeffs_1z = np.array(one_zone['scaling_analysis']['coefficients'])
fids_1z = np.array(one_zone['scaling_analysis']['fidelities'])

coeffs_2z = np.array(two_zone['scaling_analysis']['coefficients'])
fids_2z = np.array(two_zone['scaling_analysis']['fidelities'])

# Exponential decay fit
def exp_decay(x, A, b):
    return A * np.exp(-b * x)

# Create plot
fig, ax = plt.subplots(figsize=(10, 6))

# OneZone (red)
ax.scatter(coeffs_1z, fids_1z, s=70, alpha=0.8, color='#e74c3c',
           edgecolors='black', linewidth=0.5, marker='o', label='OneZone (data)')

# TwoZone (blue)
ax.scatter(coeffs_2z, fids_2z, s=70, alpha=0.8, color='#3498db',
           edgecolors='black', linewidth=0.5, marker='s', label='TwoZone (data)')

# Fit and plot curves
try:
    popt_1z, _ = curve_fit(exp_decay, coeffs_1z, fids_1z, p0=[0.9, 0.5], maxfev=10000)
    fit_x = np.linspace(0.1, 3.0, 100)
    ax.plot(fit_x, exp_decay(fit_x, *popt_1z), '-', color='#e74c3c', linewidth=2,
            label=f'OneZone: {popt_1z[0]:.2f}·e^(-{popt_1z[1]:.2f}x)')
    print(f"OneZone fit: A={popt_1z[0]:.4f}, b={popt_1z[1]:.4f}")
except Exception as e:
    print(f"OneZone fit failed: {e}")

try:
    popt_2z, _ = curve_fit(exp_decay, coeffs_2z, fids_2z, p0=[0.9, 0.5], maxfev=10000)
    fit_x = np.linspace(0.1, 3.0, 100)
    ax.plot(fit_x, exp_decay(fit_x, *popt_2z), '-', color='#3498db', linewidth=2,
            label=f'TwoZone: {popt_2z[0]:.2f}·e^(-{popt_2z[1]:.2f}x)')
    print(f"TwoZone fit: A={popt_2z[0]:.4f}, b={popt_2z[1]:.4f}")
except Exception as e:
    print(f"TwoZone fit failed: {e}")

# Formatting
ax.set_xlabel('Scaling Coefficient', fontsize=12, fontweight='bold')
ax.set_ylabel('Fidelity', fontsize=12, fontweight='bold')
ax.set_title('Scaling Coefficient: OneZone vs TwoZone Noise Model', fontsize=14, fontweight='bold')
ax.set_ylim([0, 1.0])
ax.set_xlim([0, 3.1])
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend(fontsize=10, loc='upper right')

plt.tight_layout()
plt.savefig('scaling_onezone_vs_twozone.png', dpi=150, bbox_inches='tight')
print("✓ Saved: scaling_onezone_vs_twozone.png")
plt.show()
