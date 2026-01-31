## implement fig A4
import numpy as np
import stim
from math import pi


def injection_circuit():
    """Build the injection circuit directly in stim."""
    circuit = stim.Circuit()
    
    # Prepare magic state on qubit 6 (identity operation for |0⟩)
    # Apply RY(-π/2) to qubits 0-5 using R gate (rotation around arbitrary axis)
    # In stim, we use the general rotation. For now, we'll use XCZ gates to approximate
    # or use stim's built-in gates. Stim is Clifford + T, so we approximate RY.
    
    # For a Clifford approximation, RY(-π/2) ≈ H·S·H (up to phases)
    # Using H gates as approximation for RY gates:
    for q in range(6):
        circuit.append("H", [q])
    
    # CZ gates: (1,2), (3,4), (5,6)
    circuit.append("CZ", [1, 2])
    circuit.append("CZ", [3, 4])
    circuit.append("CZ", [5, 6])
    
    # H on qubit 6
    circuit.append("H", [6])
    
    # CZ gates: (0,3), (2,5), (4,6)
    circuit.append("CZ", [0, 3])
    circuit.append("CZ", [2, 5])
    circuit.append("CZ", [4, 6])
    
    # H on qubits 2,3,4,5,6
    for q in [2, 3, 4, 5, 6]:
        circuit.append("H", [q])
    
    # CZ gates: (0,1), (2,3), (4,5)
    circuit.append("CZ", [0, 1])
    circuit.append("CZ", [2, 3])
    circuit.append("CZ", [4, 5])
    
    # H on qubits 1,2,4
    for q in [1, 2, 4]:
        circuit.append("H", [q])
    
    return circuit


def encode_a4_circuit():
    """Build the full encode_a4 circuit directly in stim."""
    circuit = stim.Circuit()
    
    # Add injection circuit to first 7 qubits (0-6)
    injection = injection_circuit()
    circuit += injection
    
    # Initialize H on qubits 8, 12, 13
    circuit.append("H", [8])
    circuit.append("H", [12])
    circuit.append("H", [13])
    
    # CNOT gates with various controls and targets
    cnot_gates = [
        (8, 4), (6, 10), (5, 10), (8, 10), (8, 0),
        (4, 9), (1, 10), (8, 2), (3, 9), (6, 10),
        (8, 9), (8, 6), (6, 9),
        (4, 11), (12, 6), (13, 5), (13, 11), (0, 11),
        (12, 4), (13, 1), (2, 11), (12, 3), (13, 6),
        (12, 11), (6, 11), (12, 5), (13, 2),
    ]
    
    for control, target in cnot_gates:
        circuit.append("CNOT", [control, target])
    
    # H on qubits 8, 12, 13 again
    circuit.append("H", [8])
    circuit.append("H", [12])
    circuit.append("H", [13])
    
    # Measure all 14 qubits
    circuit.append("M", list(range(14)))
    
    return circuit


# Build and sample the circuit
stim_enc = encode_a4_circuit()
sampler = stim_enc.compile_sampler()
samples_enc = np.array(sampler.sample(shots=500))

print("MSD/Steane encoding |0⟩ -> |0_L⟩, then measure all 14 qubits.")
print("Circuit depth:", len(stim_enc))
print("Sample shape:", samples_enc.shape)
print("First 5 shots:\n", samples_enc[:5])