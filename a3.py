##########################
## implement a3 in squin ##
##########################
from bloqade import squin
from bloqade.types import Qubit
from kirin.dialects import ilist
import numpy as np
from math import pi
from typing import Literal


# ---------------------------------------------------------------------------
# Steane [[7,1,3]] Encoding Functions
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# A3 Circuit: Fault-tolerant Steane syndrome extraction
# ---------------------------------------------------------------------------
@squin.kernel
def a3_circuit():
    """A3 circuit implementation in squin."""
    q = squin.qalloc(15)
    
    # Qubits 0-6: logical data
    # Qubits 8-14: ancilla block

    # Prepare ancilla as |+⟩_L for first half (X-stabilizer syndrome)
    steane_encode_plus_on(q[8:15])

    # CNOTs: data → ancilla (X-syndrome extraction)
    for i in range(7):
        squin.cx(q[i], q[i+8])
    
    # Measure ancilla qubits 8-14 (X-syndrome bits)
    for i in range(8, 15):
        squin.measure(q[i])

    # Reset ancilla qubits 8-14
    for i in range(8, 15):
        squin.reset(q[i])
    
    # Prepare ancilla as |0⟩_L for second half (Z-stabilizer syndrome)
    steane_encode_zero_on(q[8:15])

    # CNOTs: ancilla → data (Z-syndrome extraction)
    for i in range(7):
        squin.cx(q[i+8], q[i])

    # Transversal H on ancilla
    for i in range(7):
        squin.h(q[i+8])
    
    # Measure ancilla qubits 8-14 (Z-syndrome bits)
    for i in range(8, 15):
        squin.measure(q[i])

    return squin.result()

stim_enc = a3_circuit()
sampler = stim_enc.compile_sampler()
samples_enc = np.array(sampler.sample(shots=500))

print("MSD/Steane encoding |0⟩ -> |0_L⟩, then measure all 14 qubits.")
print("Circuit depth:", len(stim_enc))
print("Sample shape:", samples_enc.shape)
print("First 5 shots:\n", samples_enc[:5])