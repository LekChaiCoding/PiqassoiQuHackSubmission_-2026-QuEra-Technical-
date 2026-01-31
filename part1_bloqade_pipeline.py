"""
Part I: Basic pipeline using Bloqade Squin kernels only.

All circuits are defined as Squin kernels. We emit them to Cirq (via emit_circuit),
then convert back to Squin (load_circuit) for Stim/Tsim simulation.

- ghz_ideal_3: 3-qubit GHZ state, no noise
- ghz_manual_noise_3: 3-qubit GHZ with depolarizing noise (p1, p2)
- Heuristic noise: emit ideal GHZ to Cirq, apply GeminiOneZoneNoiseModel, load -> Stim
"""

from __future__ import annotations

import numpy as np
from bloqade import squin

import bloqade.stim
import bloqade.tsim
from bloqade.cirq_utils import noise
from bloqade.cirq_utils.emit import emit_circuit
from bloqade.cirq_utils import load_circuit


# ---------------------------------------------------------------------------
# 1. Squin kernel: ideal 3-qubit GHZ
# ---------------------------------------------------------------------------


@squin.kernel
def ghz_ideal_3():
    """Prepare (|000⟩ + |111⟩)/√2 and measure. All gates in Squin."""
    q = squin.qalloc(3)
    squin.h(q[0])
    squin.cx(q[0], q[1])
    squin.cx(q[1], q[2])
    squin.broadcast.measure(q)


# ---------------------------------------------------------------------------
# 2. Squin kernel: GHZ with manual depolarizing noise
# ---------------------------------------------------------------------------


@squin.kernel
def ghz_manual_noise_3(p1: float, p2: float):
    """3-qubit GHZ with 1-qubit depolarize(p1) and 2-qubit depolarize2(p2) after gates."""
    q = squin.qalloc(3)
    squin.h(q[0])
    squin.depolarize(p1, q[0])
    squin.cx(q[0], q[1])
    squin.depolarize2(p2, q[0], q[1])
    squin.cx(q[1], q[2])
    squin.depolarize2(p2, q[1], q[2])
    squin.depolarize(p1, q[1])
    squin.depolarize(p1, q[2])
    squin.broadcast.measure(q)


# ---------------------------------------------------------------------------
# 3. Run ideal: emit kernel -> Cirq -> load_circuit -> Squin -> Stim/Tsim
# ---------------------------------------------------------------------------


def run_ideal_stim(shots: int = 1000) -> np.ndarray:
    """Ideal GHZ: Squin kernel -> emit_circuit -> load_circuit -> Stim."""
    cirq_circ = emit_circuit(ghz_ideal_3)
    squin_circ = load_circuit(cirq_circ)
    stim_circ = bloqade.stim.Circuit(squin_circ)
    sampler = stim_circ.compile_sampler()
    return np.array(sampler.sample(shots=shots))


def run_ideal_tsim(shots: int = 1000) -> np.ndarray:
    """Ideal GHZ: Squin kernel -> emit_circuit -> load_circuit -> Tsim."""
    cirq_circ = emit_circuit(ghz_ideal_3)
    squin_circ = load_circuit(cirq_circ)
    tsim_circ = bloqade.tsim.Circuit(squin_circ)
    sampler = tsim_circ.compile_sampler()
    return np.array(sampler.sample(shots=shots))


# ---------------------------------------------------------------------------
# 4. Manual noise: kernel with p1, p2 -> emit_circuit(kernel, args=(p1, p2)) -> ...
# ---------------------------------------------------------------------------


def run_manual_noise_stim(
    shots: int = 1000,
    p1: float = 0.01,
    p2: float = 0.01,
) -> np.ndarray:
    """GHZ with manual noise: Squin kernel with args -> emit -> load -> Stim."""
    cirq_circ = emit_circuit(ghz_manual_noise_3, args=(p1, p2))
    squin_circ = load_circuit(cirq_circ)
    stim_circ = bloqade.stim.Circuit(squin_circ)
    sampler = stim_circ.compile_sampler()
    return np.array(sampler.sample(shots=shots))


# ---------------------------------------------------------------------------
# 5. Heuristic noise: ideal kernel -> Cirq -> noise model -> load -> Stim
# ---------------------------------------------------------------------------


def run_heuristic_noise_stim(
    shots: int = 1000,
    scaling_factor: float = 1.0,
) -> np.ndarray:
    """Emit ideal GHZ kernel to Cirq, apply GeminiOneZoneNoiseModel, load -> Stim."""
    cirq_circ = emit_circuit(ghz_ideal_3)
    noise_model = noise.GeminiOneZoneNoiseModel(scaling_factor=scaling_factor)
    cirq_noisy = noise.transform_circuit(cirq_circ, model=noise_model)
    squin_noisy = load_circuit(cirq_noisy)
    stim_circ = bloqade.stim.Circuit(squin_noisy)
    sampler = stim_circ.compile_sampler()
    return np.array(sampler.sample(shots=shots))


# ---------------------------------------------------------------------------
# 6. Analysis and diagram
# ---------------------------------------------------------------------------


def ghz_parity_fraction(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    all_zero = (samples == 0).all(axis=1)
    all_one = (samples == 1).all(axis=1)
    return np.logical_or(all_zero, all_one).mean()


def report(name: str, samples: np.ndarray) -> None:
    parity = ghz_parity_fraction(samples)
    print(f"  {name}: shots={samples.shape[0]}, parity_ok={parity:.4f}, shape={samples.shape}")


def main() -> None:
    shots = 2000

    print("Part I: Bloqade pipeline (Squin kernels only)")
    print("=" * 60)

    print("\n1. Ideal GHZ (Stim) — from ghz_ideal_3 kernel")
    ideal_stim = run_ideal_stim(shots=shots)
    report("ideal_stim", ideal_stim)

    print("\n2. Ideal GHZ (Tsim) — from ghz_ideal_3 kernel")
    ideal_tsim = run_ideal_tsim(shots=shots)
    report("ideal_tsim", ideal_tsim)

    print("\n3. Manual noise (p1=0.02, p2=0.01) — from ghz_manual_noise_3 kernel")
    manual = run_manual_noise_stim(shots=shots, p1=0.02, p2=0.01)
    report("manual_noise", manual)

    print("\n4. Heuristic noise (scaling=0.1) — ghz_ideal_3 -> Cirq -> noise -> Stim")
    heuristic = run_heuristic_noise_stim(shots=shots, scaling_factor=0.1)
    report("heuristic_noise", heuristic)

    print("\n5. Tsim circuit diagram (from ghz_ideal_3)")
    cirq_diagram = emit_circuit(ghz_ideal_3)
    squin_for_diagram = load_circuit(cirq_diagram)
    tsim_circ = bloqade.tsim.Circuit(squin_for_diagram)
    try:
        fig = tsim_circ.diagram(height=300)
        if hasattr(fig, "write_html"):
            fig.write_html("part1_ghz_diagram.html")
            print("  Diagram saved to part1_ghz_diagram.html")
        else:
            print("  Diagram object created (display in notebook)")
    except Exception as e:
        print(f"  Diagram: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
