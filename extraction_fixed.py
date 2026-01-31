# Fixed extraction cell — paste this into your PiqassoPartII notebook (Section 3. Extraction).
# Fixes: (1) squin.cx(q[3], q[5]) not q[3,q[5]]; (2) Literal[7] not Literal[8]; (3) no q[7]; (4) encode_0 = Steane |0⟩_L; (5) encode_plus = |+⟩_L, no measure.

@squin.kernel
def encode_0(q: ilist.IList[Qubit, Literal[7]]):
    """Steane [[7,1,3]]: encode |0⟩^7 -> |0⟩_L."""
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
def encode_plus(q: ilist.IList[Qubit, Literal[7]]):
    """Steane [[7,1,3]]: encode 7 qubits as |+⟩_L (|0⟩_L then transversal H)."""
    encode_0(q)
    for i in range(7):
        squin.h(q[i])


@squin.kernel
def a3extraction(data: ilist.IList[Qubit, Literal[7]], ancilla: ilist.IList[Qubit, Literal[7]]):
    """Full fault-tolerant Steane EC cycle: 14 qubits (7 data + 7 ancilla). First half: ancilla |+⟩_L, CNOT data→ancilla, measure (X-syndrome), reset. Second half: ancilla |0⟩_L, CNOT ancilla→data, H on ancilla, measure (Z-syndrome)."""
    encode_plus(ancilla)
    for i in range(7):
        squin.cx(data[i], ancilla[i])
    squin.broadcast.measure(ancilla)
    squin.broadcast.reset(ancilla)
    encode_0(ancilla)
    for i in range(7):
        squin.cx(ancilla[i], data[i])
    for i in range(7):
        squin.h(ancilla[i])
    squin.broadcast.measure(ancilla)


@squin.kernel
def executiona3():
    """Sequential: (1) inject magic state into 7 data qubits; (2) run a3 extraction on that data + 7 ancillas."""
    q = squin.qalloc(7)
    injection(q)
    ancilla = squin.qalloc(7)
    a3extraction(q, ancilla)
