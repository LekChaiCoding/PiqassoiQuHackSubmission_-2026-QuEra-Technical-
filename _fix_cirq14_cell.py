import json

path = "PiqassoPartII.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Replace cell 62 (0-based index) with 4-pair version
new_source = """# cirq14: run A4 circuit once, then analyze each of the 4 flag ancillas separately (4 pairs of graphs)
CIRQ14_SHOTS = 1000
cirq14 = emit_circuit(executiona4)
squin_cirq14 = load_circuit(cirq14)
stim_cirq14 = bloqade.stim.Circuit(squin_cirq14)
sampler_cirq14 = stim_cirq14.compile_sampler()
samples_cirq14 = np.array(sampler_cirq14.sample(shots=CIRQ14_SHOTS))

# Sample layout: 6 columns - ancilla[0]..ancilla[5]. Flag qubits = columns 1, 2, 4, 5
FLAG_ANCILLAS_CIRQ14 = [1, 2, 4, 5]  # ancilla indices = column indices

for col in FLAG_ANCILLAS_CIRQ14:
    flag_col = samples_cirq14[:, col]
    n_kept = np.sum(flag_col == 0)
    n_discarded = CIRQ14_SHOTS - n_kept
    discard_rate = n_discarded / CIRQ14_SHOTS
    print(f"cirq14 ancilla[{col}]:  Shots: {CIRQ14_SHOTS}  |  Kept: {n_kept}  |  Discarded: {n_discarded}  |  Discard rate: {discard_rate:.4f}")
    plot_kept_vs_discarded(n_kept, CIRQ14_SHOTS, title=f"2.6.1.7 cirq14 ancilla[{col}]: kept vs not kept", kept_label="Kept\\n(flag=0)", discarded_label="Discarded\\n(flag=1)")
    plot_flag_outcome_binary(n_kept, n_discarded, title=f"Flag outcome distribution (cirq14 ancilla[{col}])")
"""

# Notebook source: list of strings, each line typically ends with \n
lines = [line + "\n" for line in new_source.strip().split("\n")]

# Ensure last line has newline
if lines and not lines[-1].endswith("\n"):
    lines[-1] += "\n"

nb["cells"][62]["source"] = lines
nb["cells"][62]["outputs"] = []  # clear old outputs so they don't show stale results
if "execution_count" in nb["cells"][62]:
    del nb["cells"][62]["execution_count"]

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)
print("Updated cell 62 (cirq14 code cell) with 4-pair version. Done.")
