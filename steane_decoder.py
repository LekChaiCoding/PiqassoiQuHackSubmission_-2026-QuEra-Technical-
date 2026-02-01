"""
================================================================================
FLAG-AWARE DECODER AND VERIFICATION FOR STEANE [[7,1,3]] CODE
================================================================================
A4 Flagged Syndrome Extraction - Complete Implementation

This module provides:
1. Measurement parsing for flag and syndrome bits
2. Flag-aware lookup table decoder
3. Monte Carlo simulation for A3 vs A4 comparison
4. Integration helpers for Bloqade/Stim

Author: Your Name
Date: 2024
================================================================================
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Callable
from collections import defaultdict
from itertools import combinations
import stim

# ==============================================================================
# STEANE [[7,1,3]] CODE DEFINITIONS
# ==============================================================================

# Stabilizer generators (support as qubit indices)
# X-type stabilizers (detect Z errors)
SX_SUPPORTS = [
    [0, 1, 2, 3],  # SX1
    [0, 1, 4, 5],  # SX2
    [0, 2, 4, 6],  # SX3
]

# Z-type stabilizers (detect X errors)
SZ_SUPPORTS = [
    [0, 1, 2, 3],  # SZ1
    [0, 1, 4, 5],  # SZ2
    [0, 2, 4, 6],  # SZ3
]

# Stabilizers as binary vectors (length 7)
X_STABILIZERS = np.array([
    [1, 1, 1, 1, 0, 0, 0],  # SX1: qubits 0,1,2,3
    [1, 1, 0, 0, 1, 1, 0],  # SX2: qubits 0,1,4,5
    [1, 0, 1, 0, 1, 0, 1],  # SX3: qubits 0,2,4,6
], dtype=np.int8)

Z_STABILIZERS = np.array([
    [1, 1, 1, 1, 0, 0, 0],  # SZ1: qubits 0,1,2,3
    [1, 1, 0, 0, 1, 1, 0],  # SZ2: qubits 0,1,4,5
    [1, 0, 1, 0, 1, 0, 1],  # SZ3: qubits 0,2,4,6
], dtype=np.int8)

# Logical operators
X_LOGICAL = np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.int8)  # X on all 7 qubits
Z_LOGICAL = np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.int8)  # Z on all 7 qubits


# ==============================================================================
# MEASUREMENT PARSING
# ==============================================================================

class MeasurementParser:
    """
    Parse measurement records from A4 flagged syndrome extraction.
    
    Measurement order for A4 (your circuit):
        Block 1: SX(1), SZ(2), SZ(3) - each has [flag, syndrome]
        Block 2: SZ(1), SX(2), SX(3) - each has [flag, syndrome]
    
    Total: 12 measurements (6 flags + 6 syndromes)
    """
    
    def __init__(self, measurement_order: str = "interleaved"):
        """
        Args:
            measurement_order: How measurements are ordered
                - "interleaved": [flag0, syn0, flag1, syn1, ...]
                - "grouped": [flag0, flag1, ..., syn0, syn1, ...]
                - "syndrome_first": [syn0, flag0, syn1, flag1, ...]
        """
        self.measurement_order = measurement_order
    
    def parse(self, record: np.ndarray, num_stabilizers: int = 6
              ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse measurement record into flags and syndromes.
        
        Args:
            record: Raw measurement record from Stim
            num_stabilizers: Number of stabilizers (default 6 for Steane)
        
        Returns:
            (flags, syndromes) as numpy arrays
        """
        flags = np.zeros(num_stabilizers, dtype=np.int8)
        syndromes = np.zeros(num_stabilizers, dtype=np.int8)
        
        if self.measurement_order == "interleaved":
            # [flag0, syn0, flag1, syn1, ...]
            for i in range(num_stabilizers):
                flags[i] = record[2 * i]
                syndromes[i] = record[2 * i + 1]
                
        elif self.measurement_order == "grouped":
            # [flag0, flag1, ..., syn0, syn1, ...]
            flags = record[:num_stabilizers].astype(np.int8)
            syndromes = record[num_stabilizers:2*num_stabilizers].astype(np.int8)
            
        elif self.measurement_order == "syndrome_first":
            # [syn0, flag0, syn1, flag1, ...]
            for i in range(num_stabilizers):
                syndromes[i] = record[2 * i]
                flags[i] = record[2 * i + 1]
        
        return flags, syndromes
    
    def parse_a3_steane(self, record: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse A3 Steane-style syndrome extraction (no flags).
        
        Your A3 circuit measures qubits 7-20 (14 measurements):
            - Qubits 7-13: X-syndrome ancilla (7 bits)
            - Qubits 14-20: Z-syndrome ancilla (7 bits)
        
        Returns:
            (x_syndrome_raw, z_syndrome_raw) - 7 bits each
        """
        x_syndrome_raw = record[0:7].astype(np.int8)   # First 7 measurements
        z_syndrome_raw = record[7:14].astype(np.int8)  # Next 7 measurements
        return x_syndrome_raw, z_syndrome_raw
    
    def decode_steane_syndrome(self, x_raw: np.ndarray, z_raw: np.ndarray
                                ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert raw 7-bit ancilla measurements to 3-bit syndromes.
        
        For Steane extraction, the logical ancilla encodes the syndrome.
        The 3-bit syndrome is extracted via parity checks on the 7 bits.
        """
        # Parity check matrix for extracting syndrome from ancilla
        H = np.array([
            [1, 0, 1, 0, 1, 0, 1],  # Bit 0
            [0, 1, 1, 0, 0, 1, 1],  # Bit 1
            [0, 0, 0, 1, 1, 1, 1],  # Bit 2
        ], dtype=np.int8)
        
        syn_x = (H @ x_raw) % 2  # 3-bit X syndrome
        syn_z = (H @ z_raw) % 2  # 3-bit Z syndrome
        
        return syn_x, syn_z


# ==============================================================================
# SYNDROME COMPUTATION
# ==============================================================================

def compute_syndrome(error_x: np.ndarray, error_z: np.ndarray
                     ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Compute syndrome from X and Z error patterns.
    
    Args:
        error_x: Binary vector of X errors (length 7)
        error_z: Binary vector of Z errors (length 7)
    
    Returns:
        (syn_x, syn_z): X-syndrome (from Z errors) and Z-syndrome (from X errors)
    """
    # X errors anticommute with Z stabilizers -> trigger Z syndrome
    syn_z = tuple((Z_STABILIZERS @ error_x) % 2)
    # Z errors anticommute with X stabilizers -> trigger X syndrome
    syn_x = tuple((X_STABILIZERS @ error_z) % 2)
    
    return syn_x, syn_z


def syndrome_to_int(syn_x: Tuple[int, ...], syn_z: Tuple[int, ...]
                    ) -> int:
    """Convert syndrome tuples to single integer for LUT indexing."""
    combined = syn_x + syn_z
    result = 0
    for i, bit in enumerate(combined):
        result |= (bit << i)
    return result


def int_to_syndrome(val: int, num_bits: int = 6
                    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Convert integer back to syndrome tuples."""
    bits = tuple((val >> i) & 1 for i in range(num_bits))
    return bits[:3], bits[3:]


# ==============================================================================
# LOOKUP TABLE DECODER
# ==============================================================================

class SteaneDecoder:
    """
    Lookup table decoder for Steane [[7,1,3]] code.
    
    Supports both standard (A3) and flag-aware (A4) decoding.
    """
    
    def __init__(self):
        self.standard_lut: Dict[Tuple, Tuple[np.ndarray, np.ndarray]] = {}
        self.flag_lut: Dict[Tuple, Tuple[np.ndarray, np.ndarray]] = {}
        self._build_standard_lut()
        self._build_flag_lut()
    
    def _build_standard_lut(self):
        """Build standard syndrome -> correction lookup table."""
        # No error
        self.standard_lut[(0, 0, 0, 0, 0, 0)] = (
            np.zeros(7, dtype=np.int8),
            np.zeros(7, dtype=np.int8)
        )
        
        # Single-qubit X errors
        for q in range(7):
            error_x = np.zeros(7, dtype=np.int8)
            error_x[q] = 1
            syn_x, syn_z = compute_syndrome(error_x, np.zeros(7, dtype=np.int8))
            key = syn_x + syn_z
            self.standard_lut[key] = (error_x.copy(), np.zeros(7, dtype=np.int8))
        
        # Single-qubit Z errors
        for q in range(7):
            error_z = np.zeros(7, dtype=np.int8)
            error_z[q] = 1
            syn_x, syn_z = compute_syndrome(np.zeros(7, dtype=np.int8), error_z)
            key = syn_x + syn_z
            if key not in self.standard_lut:
                self.standard_lut[key] = (np.zeros(7, dtype=np.int8), error_z.copy())
        
        # Single-qubit Y errors (X and Z on same qubit)
        for q in range(7):
            error_x = np.zeros(7, dtype=np.int8)
            error_z = np.zeros(7, dtype=np.int8)
            error_x[q] = 1
            error_z[q] = 1
            syn_x, syn_z = compute_syndrome(error_x, error_z)
            key = syn_x + syn_z
            if key not in self.standard_lut:
                self.standard_lut[key] = (error_x.copy(), error_z.copy())
    
    def _build_flag_lut(self):
        """
        Build flag-aware LUT for A4 protocol.
        
        Hook errors in weight-4 stabilizers can cause weight-2 data errors.
        The flag qubit detects these, allowing correct decoding.
        
        For each stabilizer measurement with CNOT order [q0, q1, q2, q3]:
            - Error after CNOT to q0, q1 propagates to q2, q3 -> weight-2 error
            - Flag CNOT placed after q1 detects this
        """
        # Copy standard LUT entries with flag=False
        for syn, correction in self.standard_lut.items():
            self.flag_lut[(syn, (0, 0, 0, 0, 0, 0))] = correction
        
        # Define hook errors for each stabilizer
        # Format: stabilizer_idx -> list of (q1, q2) weight-2 errors when flagged
        # These depend on your CNOT ordering in the flag circuit
        
        # For SX stabilizers (causes Z errors on data via hook)
        # For SZ stabilizers (causes X errors on data via hook)
        
        # A4 Block 1: SX(1), SZ(2), SZ(3)
        # A4 Block 2: SZ(1), SX(2), SX(3)
        
        # Hook errors when flag triggers (adjust based on your CNOT order)
        HOOK_ERRORS_X = {
            # SZ stabilizers can cause X hook errors
            # stabilizer_idx: [(q1, q2), ...] pairs that get X errors
            1: [(2, 3)],  # SZ2 on qubits 0,1,4,5 - hook hits later qubits
            2: [(4, 6)],  # SZ3 on qubits 0,2,4,6
            3: [(2, 3)],  # SZ1 on qubits 0,1,2,3
        }
        
        HOOK_ERRORS_Z = {
            # SX stabilizers can cause Z hook errors
            0: [(2, 3)],  # SX1 on qubits 0,1,2,3
            4: [(4, 5)],  # SX2 on qubits 0,1,4,5
            5: [(4, 6)],  # SX3 on qubits 0,2,4,6
        }
        
        # Add X hook errors to flag LUT
        for stab_idx, qubit_pairs in HOOK_ERRORS_X.items():
            for q1, q2 in qubit_pairs:
                error_x = np.zeros(7, dtype=np.int8)
                error_x[q1] = 1
                error_x[q2] = 1
                syn_x, syn_z = compute_syndrome(error_x, np.zeros(7, dtype=np.int8))
                syndrome = syn_x + syn_z
                
                # Flag pattern: only this stabilizer's flag is set
                flag_pattern = tuple(1 if i == stab_idx else 0 for i in range(6))
                
                self.flag_lut[(syndrome, flag_pattern)] = (
                    error_x.copy(),
                    np.zeros(7, dtype=np.int8)
                )
        
        # Add Z hook errors to flag LUT
        for stab_idx, qubit_pairs in HOOK_ERRORS_Z.items():
            for q1, q2 in qubit_pairs:
                error_z = np.zeros(7, dtype=np.int8)
                error_z[q1] = 1
                error_z[q2] = 1
                syn_x, syn_z = compute_syndrome(np.zeros(7, dtype=np.int8), error_z)
                syndrome = syn_x + syn_z
                
                flag_pattern = tuple(1 if i == stab_idx else 0 for i in range(6))
                
                self.flag_lut[(syndrome, flag_pattern)] = (
                    np.zeros(7, dtype=np.int8),
                    error_z.copy()
                )
    
    def decode_standard(self, syndrome: Tuple[int, ...]
                        ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Decode using standard LUT (no flag information).
        
        Args:
            syndrome: 6-bit syndrome tuple (syn_x + syn_z)
        
        Returns:
            (x_correction, z_correction) or None if decoding fails
        """
        return self.standard_lut.get(syndrome)
    
    def decode_with_flags(self, syndrome: Tuple[int, ...], 
                          flags: Tuple[int, ...]
                          ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Decode using flag-aware LUT.
        
        Args:
            syndrome: 6-bit syndrome tuple
            flags: 6-bit flag tuple (one per stabilizer)
        
        Returns:
            (x_correction, z_correction) or None if decoding fails
        """
        # First try exact match with flag pattern
        key = (syndrome, flags)
        if key in self.flag_lut:
            return self.flag_lut[key]
        
        # If any flag is set but no exact match, try standard decode
        # (flag may have triggered but error was still weight-1)
        if any(flags):
            no_flag_key = (syndrome, (0, 0, 0, 0, 0, 0))
            if no_flag_key in self.flag_lut:
                return self.flag_lut[no_flag_key]
        
        # Fall back to standard LUT
        return self.standard_lut.get(syndrome)
    
    def decode(self, syndrome: Tuple[int, ...], 
               flags: Optional[Tuple[int, ...]] = None,
               use_flags: bool = True
               ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Main decode interface.
        
        Args:
            syndrome: 6-bit syndrome
            flags: 6-bit flags (optional)
            use_flags: Whether to use flag information
        
        Returns:
            (x_correction, z_correction) or None
        """
        if use_flags and flags is not None:
            return self.decode_with_flags(syndrome, flags)
        return self.decode_standard(syndrome)


# ==============================================================================
# ERROR INJECTION AND NOISY SIMULATION
# ==============================================================================

class NoisySimulator:
    """
    Monte Carlo simulator for noisy syndrome extraction.
    """
    
    def __init__(self, physical_error_rate: float, seed: Optional[int] = None):
        """
        Args:
            physical_error_rate: Probability of error per gate/location
            seed: Random seed for reproducibility
        """
        self.p = physical_error_rate
        self.rng = np.random.default_rng(seed)
    
    def depolarizing_error(self, num_qubits: int = 7
                           ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply depolarizing noise to data qubits.
        
        Each qubit independently gets I, X, Y, or Z with probabilities
        (1-p, p/3, p/3, p/3).
        
        Returns:
            (error_x, error_z) binary vectors
        """
        error_x = np.zeros(num_qubits, dtype=np.int8)
        error_z = np.zeros(num_qubits, dtype=np.int8)
        
        for q in range(num_qubits):
            r = self.rng.random()
            if r < self.p / 3:
                error_x[q] = 1  # X error
            elif r < 2 * self.p / 3:
                error_z[q] = 1  # Z error
            elif r < self.p:
                error_x[q] = 1  # Y error
                error_z[q] = 1
            # else: no error (I)
        
        return error_x, error_z
    
    def single_qubit_error(self, qubit: int, num_qubits: int = 7
                           ) -> Tuple[np.ndarray, np.ndarray]:
        """Inject a random single-qubit error at specified location."""
        error_x = np.zeros(num_qubits, dtype=np.int8)
        error_z = np.zeros(num_qubits, dtype=np.int8)
        
        r = self.rng.random()
        if r < 1/3:
            error_x[qubit] = 1
        elif r < 2/3:
            error_z[qubit] = 1
        else:
            error_x[qubit] = 1
            error_z[qubit] = 1
        
        return error_x, error_z
    
    def hook_error(self, stabilizer_idx: int, cnot_order: List[int]
                   ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Simulate a hook error during stabilizer measurement.
        
        Returns:
            (error_x, error_z, flag_triggered)
        """
        error_x = np.zeros(7, dtype=np.int8)
        error_z = np.zeros(7, dtype=np.int8)
        flag = False
        
        if self.rng.random() < self.p:
            # Error occurred - decide where in CNOT sequence
            error_location = self.rng.integers(0, len(cnot_order))
            
            # Flag is placed after position 1 (between halves)
            flag = error_location < 2
            
            # Error propagates to remaining qubits
            for q in cnot_order[error_location:]:
                # For X-stabilizer: causes Z errors; for Z-stabilizer: causes X errors
                if stabilizer_idx < 3:  # X-type
                    error_z[q] = 1
                else:  # Z-type
                    error_x[q] = 1
        
        return error_x, error_z, flag
    
    def simulate_syndrome_extraction(self, use_a4: bool = True
                                     ) -> Tuple[np.ndarray, np.ndarray, 
                                                Tuple[int, ...], Tuple[int, ...]]:
        """
        Simulate one round of noisy syndrome extraction.
        
        Args:
            use_a4: Whether to use A4 (flagged) or A3 (unflagged) protocol
        
        Returns:
            (data_error_x, data_error_z, syndrome, flags)
        """
        # Initialize with no error
        total_error_x = np.zeros(7, dtype=np.int8)
        total_error_z = np.zeros(7, dtype=np.int8)
        flags = [0] * 6
        
        # Data preparation errors
        prep_x, prep_z = self.depolarizing_error()
        total_error_x = (total_error_x + prep_x) % 2
        total_error_z = (total_error_z + prep_z) % 2
        
        # Stabilizer measurements (6 total)
        stabilizer_orders = [
            [0, 1, 2, 3],  # SX1 / SZ1
            [0, 1, 4, 5],  # SX2 / SZ2
            [0, 2, 4, 6],  # SX3 / SZ3
            [0, 1, 2, 3],  # SZ1 / SX1
            [0, 1, 4, 5],  # SZ2 / SX2
            [0, 2, 4, 6],  # SZ3 / SX3
        ]
        
        for stab_idx, order in enumerate(stabilizer_orders):
            if use_a4:
                # A4: Hook errors can be flagged
                hook_x, hook_z, flag = self.hook_error(stab_idx, order)
                total_error_x = (total_error_x + hook_x) % 2
                total_error_z = (total_error_z + hook_z) % 2
                flags[stab_idx] = int(flag)
            else:
                # A3: No flag information
                if self.rng.random() < self.p:
                    # Simulate unflagged hook error
                    error_loc = self.rng.integers(0, len(order))
                    for q in order[error_loc:]:
                        if stab_idx < 3:
                            total_error_z[q] = (total_error_z[q] + 1) % 2
                        else:
                            total_error_x[q] = (total_error_x[q] + 1) % 2
        
        # Compute syndrome
        syn_x, syn_z = compute_syndrome(total_error_x, total_error_z)
        syndrome = syn_x + syn_z
        
        return total_error_x, total_error_z, syndrome, tuple(flags)


# ==============================================================================
# MONTE CARLO VERIFICATION
# ==============================================================================

class MonteCarloVerifier:
    """
    Monte Carlo simulation for comparing A3 vs A4 logical error rates.
    """
    
    def __init__(self, decoder: SteaneDecoder):
        self.decoder = decoder
    
    def check_logical_error(self, 
                            data_error_x: np.ndarray, 
                            data_error_z: np.ndarray,
                            correction_x: np.ndarray,
                            correction_z: np.ndarray) -> bool:
        """
        Check if residual error after correction is a logical error.
        
        Returns:
            True if logical error occurred
        """
        # Residual error
        residual_x = (data_error_x + correction_x) % 2
        residual_z = (data_error_z + correction_z) % 2
        
        # Logical X error: odd overlap of residual X with Z_logical
        logical_x = (np.dot(residual_x, Z_LOGICAL) % 2) == 1
        # Logical Z error: odd overlap of residual Z with X_logical
        logical_z = (np.dot(residual_z, X_LOGICAL) % 2) == 1
        
        return logical_x or logical_z
    
    def run_simulation(self,
                       physical_error_rate: float,
                       shots: int = 100000,
                       use_flags: bool = True,
                       seed: Optional[int] = None,
                       verbose: bool = False) -> Dict:
        """
        Run Monte Carlo simulation.
        
        Args:
            physical_error_rate: p
            shots: Number of samples
            use_flags: Use A4 (True) or A3 (False) decoding
            seed: Random seed
            verbose: Print progress
        
        Returns:
            Dict with results
        """
        simulator = NoisySimulator(physical_error_rate, seed)
        
        logical_errors = 0
        decoding_failures = 0
        flag_triggers = 0
        
        for i in range(shots):
            if verbose and (i + 1) % 10000 == 0:
                print(f"  Shot {i+1}/{shots}")
            
            # Simulate noisy extraction
            error_x, error_z, syndrome, flags = simulator.simulate_syndrome_extraction(
                use_a4=use_flags
            )
            
            if any(flags):
                flag_triggers += 1
            
            # Decode
            correction = self.decoder.decode(syndrome, flags, use_flags=use_flags)
            
            if correction is None:
                decoding_failures += 1
                logical_errors += 1  # Count as logical error
                continue
            
            corr_x, corr_z = correction
            
            # Check for logical error
            if self.check_logical_error(error_x, error_z, corr_x, corr_z):
                logical_errors += 1
        
        return {
            'physical_error_rate': physical_error_rate,
            'shots': shots,
            'logical_errors': logical_errors,
            'logical_error_rate': logical_errors / shots,
            'decoding_failures': decoding_failures,
            'flag_triggers': flag_triggers,
            'flag_trigger_rate': flag_triggers / shots,
            'use_flags': use_flags,
        }
    
    def compare_a3_vs_a4(self,
                         error_rates: List[float],
                         shots_per_rate: int = 50000,
                         seed: Optional[int] = None,
                         verbose: bool = True) -> Dict:
        """
        Compare A3 (unflagged) vs A4 (flagged) protocols.
        
        Args:
            error_rates: List of physical error rates to test
            shots_per_rate: Samples per error rate
            seed: Base random seed
            verbose: Print progress
        
        Returns:
            Dict with comparison results
        """
        results = {
            'error_rates': error_rates,
            'a3': [],
            'a4': [],
            'improvement': [],
        }
        
        for i, p in enumerate(error_rates):
            if verbose:
                print(f"\n[{i+1}/{len(error_rates)}] Testing p = {p:.2e}")
            
            # A3: Ignore flags
            if verbose:
                print("  Running A3 (no flags)...")
            a3_result = self.run_simulation(
                p, shots_per_rate, 
                use_flags=False,
                seed=seed + i if seed else None
            )
            
            # A4: Use flags
            if verbose:
                print("  Running A4 (with flags)...")
            a4_result = self.run_simulation(
                p, shots_per_rate,
                use_flags=True,
                seed=seed + i + 1000 if seed else None
            )
            
            results['a3'].append(a3_result)
            results['a4'].append(a4_result)
            
            # Compute improvement
            if a4_result['logical_error_rate'] > 0:
                improvement = a3_result['logical_error_rate'] / a4_result['logical_error_rate']
            else:
                improvement = float('inf')
            results['improvement'].append(improvement)
            
            if verbose:
                print(f"  A3 p_L = {a3_result['logical_error_rate']:.6f}")
                print(f"  A4 p_L = {a4_result['logical_error_rate']:.6f}")
                print(f"  Improvement: {improvement:.2f}x")
        
        return results


# ==============================================================================
# STIM INTEGRATION
# ==============================================================================

class StimIntegration:
    """
    Integration helpers for Bloqade/Stim circuits.
    """
    
    @staticmethod
    def build_detector_error_model(circuit: stim.Circuit) -> stim.DetectorErrorModel:
        """Extract detector error model from circuit."""
        return circuit.detector_error_model(decompose_errors=True)
    
    @staticmethod
    def run_with_tableau(circuit: stim.Circuit, shots: int = 1000
                         ) -> np.ndarray:
        """
        Run circuit using TableauSimulator (supports branching).
        
        Returns:
            Array of measurement records, shape (shots, num_measurements)
        """
        results = []
        sim = stim.TableauSimulator()
        
        for _ in range(shots):
            sim.reset_all()
            sim.do(circuit)
            record = np.array(sim.current_measurement_record(), dtype=np.int8)
            results.append(record)
        
        return np.array(results)
    
    @staticmethod
    def run_fast(circuit: stim.Circuit, shots: int = 100000,
                 seed: Optional[int] = None) -> np.ndarray:
        """
        Run circuit using fast frame simulator.
        
        Note: Does not support mid-circuit branching.
        
        Returns:
            Array of measurement records
        """
        sampler = circuit.compile_sampler(seed=seed)
        return sampler.sample(shots)
    
    @staticmethod
    def add_noise_to_circuit(circuit: stim.Circuit, 
                             p: float) -> stim.Circuit:
        """
        Add depolarizing noise after each gate.
        
        Args:
            circuit: Original circuit
            p: Error probability
        
        Returns:
            Noisy circuit
        """
        noisy = stim.Circuit()
        
        for instruction in circuit:
            noisy.append(instruction)
            
            # Add noise after gates
            if instruction.name in ['CX', 'CZ', 'CNOT']:
                targets = instruction.targets_copy()
                for i in range(0, len(targets), 2):
                    q1 = targets[i].value
                    q2 = targets[i + 1].value
                    noisy.append("DEPOLARIZE2", [q1, q2], p)
            elif instruction.name in ['H', 'X', 'Y', 'Z', 'S', 'T']:
                targets = instruction.targets_copy()
                for t in targets:
                    noisy.append("DEPOLARIZE1", [t.value], p)
        
        return noisy


# ==============================================================================
# BLOQADE INTEGRATION
# ==============================================================================

def process_bloqade_results(measurement_record: np.ndarray,
                            decoder: SteaneDecoder,
                            parser: MeasurementParser,
                            use_flags: bool = True) -> Dict:
    """
    Process measurement results from Bloqade circuit.
    
    Args:
        measurement_record: Raw measurements from bloqade.stim
        decoder: SteaneDecoder instance
        parser: MeasurementParser instance
        use_flags: Whether to use flag information
    
    Returns:
        Dict with decoded results
    """
    flags, syndromes = parser.parse(measurement_record)
    
    syndrome_tuple = tuple(syndromes)
    flag_tuple = tuple(flags)
    
    correction = decoder.decode(syndrome_tuple, flag_tuple, use_flags=use_flags)
    
    return {
        'flags': flags,
        'syndromes': syndromes,
        'syndrome_tuple': syndrome_tuple,
        'flag_tuple': flag_tuple,
        'correction': correction,
        'any_flag': any(flags),
        'any_syndrome': any(syndromes),
    }


# ==============================================================================
# VISUALIZATION AND ANALYSIS
# ==============================================================================

def print_lut(decoder: SteaneDecoder, lut_type: str = "standard"):
    """Print lookup table for inspection."""
    lut = decoder.standard_lut if lut_type == "standard" else decoder.flag_lut
    
    print(f"\n{'='*60}")
    print(f"  {lut_type.upper()} LOOKUP TABLE")
    print(f"{'='*60}")
    
    for key, (corr_x, corr_z) in sorted(lut.items(), key=lambda x: str(x[0])):
        if lut_type == "flag":
            syndrome, flags = key[:6], key[6:] if len(key) > 6 else key[1]
            print(f"Syn={syndrome}, Flags={flags}")
        else:
            print(f"Syndrome={key}")
        
        x_str = ''.join(map(str, corr_x))
        z_str = ''.join(map(str, corr_z))
        print(f"  -> X correction: {x_str} (weight {sum(corr_x)})")
        print(f"  -> Z correction: {z_str} (weight {sum(corr_z)})")
        print()


def plot_comparison(results: Dict, save_path: Optional[str] = None):
    """
    Plot A3 vs A4 comparison.
    
    Requires matplotlib (optional import).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available. Install with: pip install matplotlib")
        return
    
    error_rates = results['error_rates']
    a3_rates = [r['logical_error_rate'] for r in results['a3']]
    a4_rates = [r['logical_error_rate'] for r in results['a4']]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Log-log plot of logical error rates
    ax1.loglog(error_rates, a3_rates, 'o-', label='A3 (no flags)', markersize=8)
    ax1.loglog(error_rates, a4_rates, 's-', label='A4 (with flags)', markersize=8)
    
    # Reference lines for p^2 scaling
    p_ref = np.array(error_rates)
    ax1.loglog(p_ref, p_ref**2, '--', alpha=0.5, label=r'$p^2$ reference')
    
    ax1.set_xlabel('Physical Error Rate (p)')
    ax1.set_ylabel('Logical Error Rate')
    ax1.set_title('Logical Error Rate: A3 vs A4')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Improvement factor
    ax2.semilogx(error_rates, results['improvement'], 'o-', markersize=8, color='green')
    ax2.axhline(y=1, linestyle='--', color='gray', alpha=0.5)
    ax2.set_xlabel('Physical Error Rate (p)')
    ax2.set_ylabel('Improvement Factor (A3/A4)')
    ax2.set_title('A4 Improvement over A3')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    
    plt.show()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """
    Main execution - run A3 vs A4 comparison.
    """
    print("="*70)
    print("  STEANE [[7,1,3]] CODE: A3 vs A4 FLAGGED SYNDROME EXTRACTION")
    print("="*70)
    
    # Initialize decoder
    print("\n[1] Building lookup tables...")
    decoder = SteaneDecoder()
    print(f"    Standard LUT entries: {len(decoder.standard_lut)}")
    print(f"    Flag-aware LUT entries: {len(decoder.flag_lut)}")
    
    # Initialize verifier
    verifier = MonteCarloVerifier(decoder)
    
    # Test error rates
    error_rates = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    shots = 50000  # Increase for better statistics
    
    print(f"\n[2] Running Monte Carlo comparison...")
    print(f"    Error rates: {error_rates}")
    print(f"    Shots per rate: {shots}")
    
    # Run comparison
    results = verifier.compare_a3_vs_a4(
        error_rates=error_rates,
        shots_per_rate=shots,
        seed=42,
        verbose=True
    )
    
    # Print summary
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    print(f"\n{'p':<12} {'A3 p_L':<15} {'A4 p_L':<15} {'Improvement':<12}")
    print("-"*54)
    
    for i, p in enumerate(error_rates):
        a3_pl = results['a3'][i]['logical_error_rate']
        a4_pl = results['a4'][i]['logical_error_rate']
        imp = results['improvement'][i]
        print(f"{p:<12.2e} {a3_pl:<15.6f} {a4_pl:<15.6f} {imp:<12.2f}x")
    
    print("\n" + "="*70)
    
    # Try to plot
    print("\n[3] Generating plot...")
    plot_comparison(results, save_path="a3_vs_a4_comparison.png")
    
    return results


# ==============================================================================
# INTEGRATION WITH YOUR BLOQADE CIRCUIT
# ==============================================================================

def example_bloqade_integration():
    """
    Example showing how to integrate with your Bloqade circuit.
    
    Uncomment and modify based on your setup.
    """
    print("\n" + "="*70)
    print("  BLOQADE INTEGRATION EXAMPLE")
    print("="*70)
    
    # --- YOUR EXISTING BLOQADE CODE ---
    # from bloqade import squin
    # from bloqade.cirq_utils import load_circuit
    # from bloqade.cirq_utils.emit import emit_circuit
    # import bloqade.stim
    #
    # # Emit your circuit
    # circ = emit_circuit(a4_circuit)  # Your A4 circuit function
    # circ_load = load_circuit(circ)
    # circ_stim = bloqade.stim.Circuit(circ_load)
    #
    # # Sample measurements
    # shots = 10000
    # raw_results = circ_stim.sample(shots=shots)
    
    # --- DECODING ---
    decoder = SteaneDecoder()
    parser = MeasurementParser(measurement_order="interleaved")
    
    # --- PROCESS EACH SHOT ---
    # for shot in raw_results:
    #     result = process_bloqade_results(
    #         measurement_record=shot,
    #         decoder=decoder,
    #         parser=parser,
    #         use_flags=True
    #     )
    #     
    #     print(f"Syndrome: {result['syndrome_tuple']}")
    #     print(f"Flags: {result['flag_tuple']}")
    #     print(f"Correction: {result['correction']}")
    
    print("\nSee code comments for integration steps.")
    print("Adjust MeasurementParser order based on your circuit's measurement sequence.")


if __name__ == "__main__":
    # Run main comparison
    results = main()
    
    # Show integration example
    example_bloqade_integration()
