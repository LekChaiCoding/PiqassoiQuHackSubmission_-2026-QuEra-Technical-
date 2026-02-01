"""
Piqasso Part III: d=5 Surface Code MSD Encoding

This scales the d=3 [[7,1,3]] encoding from the original Piqasso notebook
to d=5 [[17,1,5]] encoding.

The original d=3 injection uses:
1. √Y† (RY(-π/2)) on ancilla qubits first
2. CZ layers to spread entanglement
3. √Y (RY(π/2)) corrections after CZ
4. The magic state qubit is in the middle

For d=5, we scale this pattern to 17 qubits following the lattice structure.
"""

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


class QSim:
    def __init__(self, n):
        self.n = n
        self.state = np.zeros(2**n, dtype=complex)
        self.state[0] = 1.0
    
    def apply1(self, g, q):
        new = np.zeros_like(self.state)
        for i in range(2**self.n):
            b = (i >> (self.n-1-q)) & 1
            j = i ^ (1 << (self.n-1-q))
            if b == 0:
                new[i] += g[0,0]*self.state[i] + g[0,1]*self.state[j]
            else:
                new[i] += g[1,0]*self.state[j] + g[1,1]*self.state[i]
        self.state = new
    
    def h(self, q):
        self.apply1(np.array([[1,1],[1,-1]], dtype=complex)/np.sqrt(2), q)
    
    def t(self, q):
        self.apply1(np.array([[1,0],[0,np.exp(1j*pi/4)]], dtype=complex), q)
    
    def ry(self, theta, q):
        c, s = np.cos(theta/2), np.sin(theta/2)
        self.apply1(np.array([[c,-s],[s,c]], dtype=complex), q)
    
    def cz(self, a, b):
        for i in range(2**self.n):
            if ((i >> (self.n-1-a)) & 1) and ((i >> (self.n-1-b)) & 1):
                self.state[i] *= -1
    
    def i_gate(self, q):
        """Identity - placeholder for magic state."""
        pass


def d3_injection_original():
    """
    Original d=3 injection from Piqasso Part II.
    This produces 8 non-zero states (correct for [[7,1,3]]).
    
    From the notebook:
    preparemagicstate(q[6])  # identity/T-state
    ry(-pi/2, q[0-5])        # √Y† on ancillas
    cz(1,2), cz(3,4), cz(5,6)
    ry(pi/2, q[6])
    cz(0,3), cz(2,5), cz(4,6)
    ry(pi/2, q[2,3,4,5,6])
    cz(0,1), cz(2,3), cz(4,5)
    ry(pi/2, q[1,2,4])
    """
    sim = QSim(7)
    
    # Magic state on q6
    sim.i_gate(6)  # placeholder - would be T state
    
    # Initial √Y† on ancillas
    sim.ry(-pi/2, 0)
    sim.ry(-pi/2, 1)
    sim.ry(-pi/2, 2)
    sim.ry(-pi/2, 3)
    sim.ry(-pi/2, 4)
    sim.ry(-pi/2, 5)
    
    # CZ layer 1
    sim.cz(1, 2)
    sim.cz(3, 4)
    sim.cz(5, 6)
    
    # √Y on q6
    sim.ry(pi/2, 6)
    
    # CZ layer 2
    sim.cz(0, 3)
    sim.cz(2, 5)
    sim.cz(4, 6)
    
    # √Y on multiple
    sim.ry(pi/2, 2)
    sim.ry(pi/2, 3)
    sim.ry(pi/2, 4)
    sim.ry(pi/2, 5)
    sim.ry(pi/2, 6)
    
    # CZ layer 3
    sim.cz(0, 1)
    sim.cz(2, 3)
    sim.cz(4, 5)
    
    # Final √Y
    sim.ry(pi/2, 1)
    sim.ry(pi/2, 2)
    sim.ry(pi/2, 4)
    
    return sim


def d5_injection_scaled():
    """
    d=5 [[17,1,5]] encoding scaled from d=3 [[7,1,3]].
    
    The d=5 lattice (from paper figure b):
    
              (7)  (9)
             / | \\ / |
           (6)-(8)-
          / |   |\\  
        (4)-(5)-(13)-(16)
       / |   |   |  \\/ 
     (0)-(2)-(11)-(14)
      |   |    |   / |
     (1) (3)  (10)(12)(15)
    
    Magic state goes on qubit 7 (center of lattice).
    
    We follow the same pattern as d=3:
    1. √Y† on all ancillas (qubits 0-6, 8-16)
    2. CZ layers following the lattice connectivity
    3. √Y corrections after each layer
    """
    sim = QSim(17)
    
    # Magic state on q7 (center)
    sim.h(7)
    sim.t(7)
    
    # Initial √Y† on ALL ancilla qubits (not q7)
    for q in range(17):
        if q != 7:
            sim.ry(-pi/2, q)
    
    # CZ Layer 1: Pairs connected in the lattice
    # Following pattern similar to d=3: (1,2), (3,4), (5,6) -> extended
    sim.cz(1, 3)    # bottom connections
    sim.cz(3, 10)
    sim.cz(10, 12)
    sim.cz(12, 15)
    
    sim.cz(0, 2)    # next layer up
    sim.cz(2, 11)
    sim.cz(11, 14)
    
    sim.cz(4, 5)    # middle layer
    sim.cz(5, 13)
    sim.cz(13, 16)
    
    sim.cz(6, 8)    # around center
    
    # √Y on magic state qubit
    sim.ry(pi/2, 7)
    
    # CZ Layer 2: Cross connections to center
    sim.cz(0, 4)
    sim.cz(2, 5)
    sim.cz(3, 5)
    sim.cz(5, 7)    # connect to center
    sim.cz(6, 7)
    sim.cz(7, 8)
    sim.cz(10, 11)
    sim.cz(11, 13)
    sim.cz(14, 16)
    
    # √Y corrections
    sim.ry(pi/2, 2)
    sim.ry(pi/2, 3)
    sim.ry(pi/2, 5)
    sim.ry(pi/2, 6)
    sim.ry(pi/2, 7)
    sim.ry(pi/2, 8)
    sim.ry(pi/2, 10)
    sim.ry(pi/2, 11)
    sim.ry(pi/2, 13)
    sim.ry(pi/2, 14)
    
    # CZ Layer 3: More lattice connections
    sim.cz(0, 1)
    sim.cz(2, 3)
    sim.cz(4, 6)
    sim.cz(5, 8)
    sim.cz(9, 8)
    sim.cz(10, 3)
    sim.cz(11, 10)
    sim.cz(12, 10)
    sim.cz(13, 11)
    sim.cz(14, 11)
    sim.cz(15, 12)
    sim.cz(16, 14)
    
    # Final √Y corrections
    sim.ry(pi/2, 1)
    sim.ry(pi/2, 4)
    sim.ry(pi/2, 9)
    sim.ry(pi/2, 12)
    sim.ry(pi/2, 15)
    sim.ry(pi/2, 16)
    
    return sim


# ============================================================================
# BLOQADE SQUIN VERSION  
# ============================================================================

if BLOQADE_AVAILABLE:
    
    @squin.kernel
    def prepare_magic(q):
        squin.h(q)
        squin.t(q)
    
    @squin.kernel
    def d5_injection(q: ilist.IList[Qubit, Literal[17]]):
        """d=5 encoding with magic state injection (Bloqade SQUIN)."""  
        #q = squin.qalloc(17)
        
        # Magic state on center qubit
        prepare_magic(q[7])
        
        # √Y on all ancillas
        squin.ry(pi/2, q[0])
        squin.ry(pi/2, q[1])
        squin.ry(pi/2, q[2])
        squin.ry(pi/2, q[3])
        squin.ry(pi/2, q[4])
        squin.ry(pi/2, q[5])
        squin.ry(pi/2, q[6])
        # skip q[7] - has magic state
        squin.ry(pi/2, q[8])
        squin.ry(pi/2, q[9])
        squin.ry(pi/2, q[10])
        squin.ry(pi/2, q[11])
        squin.ry(pi/2, q[12])
        squin.ry(pi/2, q[13])
        squin.ry(pi/2, q[14])
        squin.ry(pi/2, q[15])
        squin.ry(pi/2, q[16])
        
        # CZ Layer 1
        squin.cz(q[1], q[3])
        squin.cz(q[3], q[10])
        squin.cz(q[7], q[10])
        squin.cz(q[12], q[14])
        squin.cz(q[13], q[16])
        
        # √Y† on center
        squin.ry(-pi/2, q[7])
        squin.ry(-pi/2, q[16])
        
        # CZ Layer 2
        squin.cz(q[4], q[7])
        squin.cz(q[8], q[10])
        squin.cz(q[11], q[14])
        squin.cz(q[15], q[16])
        
        # √Y† layer
        squin.ry(-pi/2, q[4])
        squin.ry(-pi/2, q[10])
        squin.ry(-pi/2, q[14])
        squin.ry(-pi/2, q[16])

        
        # CZ Layer 3
        squin.cz(q[2], q[4])
        squin.cz(q[6], q[8])
        squin.cz(q[7], q[9])
        squin.cz(q[10], q[13])
        squin.cz(q[14], q[16])

        
        # Final √Y
        squin.ry(pi/2, q[1])
        squin.ry(pi/2, q[2])
        squin.ry(pi/2, q[3])
        squin.ry(pi/2, q[4])
        squin.ry(pi/2, q[6])
        squin.ry(pi/2, q[7])
        squin.ry(pi/2, q[8])
        squin.ry(pi/2, q[9])
        squin.ry(pi/2, q[11])
        squin.ry(pi/2, q[12])
        squin.ry(pi/2, q[14])

        # Final CZ 
        squin.cz(q[0], q[1])
        squin.cz(q[2], q[3])
        squin.cz(q[4], q[5])
        squin.cz(q[6], q[7])
        squin.cz(q[8], q[9])
        squin.cz(q[12], q[15])

        # Final √Y†
        squin.ry(-pi/2, q[0])
        squin.ry(-pi/2, q[2])
        squin.ry(-pi/2, q[5])
        squin.ry(-pi/2, q[6])
        squin.ry(-pi/2, q[8])
        squin.ry(-pi/2, q[10])
        squin.ry(-pi/2, q[12])



# ============================================================================
# MAIN
# ============================================================================

def analyze(state, n=17):
    probs = np.abs(state)**2
    nz = np.sum(probs > 1e-10)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Qubits: {n}")
    print(f"Non-zero: {nz}")
    print(f"Total prob: {np.sum(probs):.10f}")
    
    print(f"\nTop 10 states:")
    top = np.argsort(probs)[::-1][:10]
    for i, idx in enumerate(top):
        if probs[idx] > 1e-10:
            print(f"  {i+1}. |{format(idx, f'0{n}b')}⟩ p={probs[idx]:.6f}")
    
    return probs, nz


def main():
    print("="*60)
    print("PIQASSO PART III: d=5 ENCODING")
    print("="*60)
    
    # First verify d=3 works correctly
    print("\n--- Verifying d=3 encoding ---")
    sim3 = d3_injection_original()
    p3, nz3 = analyze(sim3.state, 7)
    print(f"Expected ~8 non-zero for d=3: {'✓' if 4 <= nz3 <= 16 else '✗'}")
    
    # Now run d=5
    print("\n--- Running d=5 encoding ---")
    
    if BLOQADE_AVAILABLE:
        cirq_c = emit_circuit(d5_injection)
        print("Emitted Successfully")
        squin_c = load_circuit(cirq_c)
        tsim_c = bloqade.tsim.Circuit(squin_c)
        state5 = np.array(tsim_c.state_vector())

            #sim5 = d5_injection_scaled()
            #state5 = sim5.state
    else:
        sim5 = d5_injection_scaled()
        state5 = sim5.state
    
    p5, nz5 = analyze(state5, 17)
    
    # Expected: d=5 should have more states than d=3 but still manageable
    # For [[17,1,5]] the logical subspace dimension depends on encoding
    print(f"\nScaling check:")
    print(f"  d=3: {nz3} non-zero states")  
    print(f"  d=5: {nz5} non-zero states")
    
    # Save plot
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # d=3 histogram
        probs3 = np.abs(sim3.state)**2
        nz_probs3 = probs3[probs3 > 1e-10]
        axes[0].bar(range(len(nz_probs3)), sorted(nz_probs3, reverse=True), color='blue', alpha=0.7)
        axes[0].set_title(f'd=3 [[7,1,3]] Encoding\n({len(nz_probs3)} non-zero states)')
        axes[0].set_xlabel('State index')
        axes[0].set_ylabel('Probability')
        
        # d=5 histogram  
        nz_probs5 = p5[p5 > 1e-10]
        if len(nz_probs5) < 200:
            axes[1].bar(range(len(nz_probs5)), sorted(nz_probs5, reverse=True), color='red', alpha=0.7)
        else:
            axes[1].hist(nz_probs5, bins=50, color='red', alpha=0.7)
        axes[1].set_title(f'd=5 [[17,1,5]] Encoding\n({len(nz_probs5)} non-zero states)')
        axes[1].set_xlabel('State index' if len(nz_probs5) < 200 else 'Probability')
        axes[1].set_ylabel('Probability' if len(nz_probs5) < 200 else 'Count')
        
        plt.suptitle('Piqasso: d=3 vs d=5 Encoding Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('/home/claude/d5_vs_d3_comparison.png', dpi=150)
        plt.close()
        print("\nPlot saved to: /home/claude/d5_vs_d3_comparison.png")
    except Exception as e:
        print(f"Could not save plot: {e}")
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
