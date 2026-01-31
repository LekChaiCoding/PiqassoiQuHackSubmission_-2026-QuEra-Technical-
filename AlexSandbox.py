"""
Piqasso Part II: MSD State Encoding & Steane QEC — Sandbox

QuEra iQuHack 2026 — Step 2. Runnable copy of PiqassoPartII.ipynb for experimentation.

Implements:
  1. MSD State Encoding circuit — Encode into [[7,1,3]] (Steane/color) code.
  2. Steane QEC — One round of syndrome extraction (Steane encoding demo with Stim).
"""

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
from typing import Literal

import numpy as np
from bloqade import squin
from bloqade.types import Qubit
from kirin.dialects import ilist
from math import pi
import bloqade.stim
import bloqade.tsim
from bloqade.cirq_utils import load_circuit
from bloqade.cirq_utils.emit import emit_circuit

# ---------------------------------------------------------------------------
# 2. Magic State Preparation and Injection (parameterized: use 7 qubits from caller)
# ---------------------------------------------------------------------------
@squin.kernel
def preparemagicstate(qubit):
    squin.i(qubit)


@squin.kernel
def injection(q: ilist.IList[Qubit, Literal[7]]):
    """Apply magic-state injection to the 7-qubit register `q` (allocated by caller)."""
    preparemagicstate(q[6])
    for i in range(6):
        squin.ry(-pi / 2, q[i])
    squin.cz(q[1], q[2])
    squin.cz(q[3], q[4])
    squin.cz(q[5], q[6])
    squin.ry(pi / 2, q[6])
    squin.cz(q[0], q[3])
    squin.cz(q[2], q[5])
    squin.cz(q[4], q[6])
    for i in range(2, 7):
        squin.ry(pi / 2, q[i])
    for i in range(0, 5, 2):
        squin.cz(q[i], q[i + 1])
    squin.ry(pi / 2, q[1])
    squin.ry(pi / 2, q[2])
    squin.ry(pi / 2, q[4])


# ---------------------------------------------------------------------------
# 3. Measure with two ancillas (Steane X-syndrome style)
# ---------------------------------------------------------------------------
@squin.kernel
def measure_with_two_ancillas(
    q: ilist.IList[Qubit, Literal[7]],
    ancillas: ilist.IList[Qubit, Literal[2]],
):
    """
    Use 2 ancillas to measure two X-type syndrome bits on the 7 data qubits.
    Ancillas are entangled with data (CNOT data -> ancilla for X syndrome), then measured.
    """
    # Ancilla 0: measure X stabilizer on qubits 1,2,4,5 (one row of Steane parity)
    squin.h(ancillas[0])
    squin.cx(ancillas[0], q[1])
    squin.cx(ancillas[0], q[2])
    squin.cx(ancillas[0], q[4])
    squin.cx(ancillas[0], q[5])
    squin.h(ancillas[0])

    # Ancilla 1: measure X stabilizer on qubits 2,3,5,6
    squin.h(ancillas[1])
    squin.cx(ancillas[1], q[2])
    squin.cx(ancillas[1], q[3])
    squin.cx(ancillas[1], q[5])
    squin.cx(ancillas[1], q[6])
    squin.h(ancillas[1])

    # Measure both ancillas (syndrome bits)
    squin.broadcast.measure(ancillas)


# ---------------------------------------------------------------------------
# 4. Full fault-tolerant Steane syndrome extraction (a3extraction)
# ---------------------------------------------------------------------------
# Steane [[7,1,3]]: encode 7 physical qubits as |0⟩_L or |+⟩_L (same CNOT pattern
# as in PiqassoPartII; |+⟩_L = encode |0⟩_L then transversal H).


@squin.kernel
def steane_encode_zero_on(q: ilist.IList[Qubit, Literal[7]]):
    """Encode |0⟩^7 → |0⟩_L on the 7-qubit register q (Steane [[7,1,3]])."""
    squin.h(q[1])
    squin.h(q[2])
    squin.h(q[3])
    squin.cx(q[1], q[4])
    squin.cx(q[2], q[4])
    squin.cx(q[1], q[5])
    squin.cx(q[3], q[5])
    squin.cx(q[2], q[6])
    squin.cx(q[3], q[6])
    squin.cx(q[4], q[0])
    squin.cx(q[5], q[0])
    squin.cx(q[6], q[0])


@squin.kernel
def steane_encode_plus_on(q: ilist.IList[Qubit, Literal[7]]):
    """Encode 7 qubits as |+⟩_L: first |0⟩_L then transversal H."""
    steane_encode_zero_on(q)
    for i in range(7):
        squin.h(q[i])


@squin.kernel
def a3extraction():
    """
    Full fault-tolerant Steane error-correction cycle on 14 qubits.

    Top 7 wires: logical data |ψ⟩_L. Lower 7 wires: ancilla block (encoded).
    - First half: ancilla encoded as |+⟩_L; transversal CNOT data→ancilla;
      measure ancilla → X-stabilizer syndrome (phase errors). Reset ancilla.
    - Second half: ancilla encoded as |0⟩_L; transversal CNOT ancilla→data;
      H on ancilla; measure ancilla → Z-stabilizer syndrome (bit-flip errors).
    """
    data = squin.qalloc(7)
    ancilla = squin.qalloc(7)

    # --- First half: X-stabilizer syndrome (Z errors on data → ancilla) ---
    steane_encode_plus_on(ancilla)
    for i in range(7):
        squin.cx(data[i], ancilla[i])
    squin.broadcast.measure(ancilla)
    squin.broadcast.reset(ancilla)

    # --- Second half: Z-stabilizer syndrome (X errors on data → ancilla) ---
    steane_encode_zero_on(ancilla)
    for i in range(7):
        squin.cx(ancilla[i], data[i])
    for i in range(7):
        squin.h(ancilla[i])
    squin.broadcast.measure(ancilla)


# ---------------------------------------------------------------------------
# 5. Main kernel: allocate 7 + 2 qubits, run injection then measure with ancillas
# ---------------------------------------------------------------------------
@squin.kernel
def injection_then_measure():
    """Allocate 7 data qubits and 2 ancillas; run injection then syndrome measurement."""
    q = squin.qalloc(7)
    ancillas = squin.qalloc(2)
    injection(q)
    measure_with_two_ancillas(q, ancillas)


# ---------------------------------------------------------------------------
# 6. Emit and run (use main kernel so all 9 qubits are in one circuit)
# ---------------------------------------------------------------------------
# Full fault-tolerant Steane extraction (14 qubits)
cirq_a3 = emit_circuit(a3extraction)
squin_a3 = load_circuit(cirq_a3)
tsim_a3 = bloqade.tsim.Circuit(squin_a3)
fig_a3 = tsim_a3.diagram(height=500)
print("a3extraction (14 qubits): fig_a3")

# Full circuit: injection + measure with 2 ancillas (9 qubits total)
cirq_full = emit_circuit(injection_then_measure)
squin_full = load_circuit(cirq_full)
tsim_full = bloqade.tsim.Circuit(squin_full)
fig_full = tsim_full.diagram(height=400)
print("Full circuit (injection + 2-ancilla measurement): 9 qubits. Diagram: fig_full")

# Standalone injection (7 qubits only) — for comparison, call from a tiny wrapper kernel
@squin.kernel
def injection_only():
    q = squin.qalloc(7)
    injection(q)


cirq_diagram = emit_circuit(injection_only)
squin_for_diagram = load_circuit(cirq_diagram)
tsim_circ = bloqade.tsim.Circuit(squin_for_diagram)
fig = tsim_circ.diagram(height=400)
print("Injection-only circuit: 7 qubits. Diagram: fig")

