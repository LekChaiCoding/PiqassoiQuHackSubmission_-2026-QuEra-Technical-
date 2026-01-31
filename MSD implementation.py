import numpy as np
from math import pi
from bloqade import squin
import bloqade.stim
import bloqade.tsim
from bloqade.cirq_utils import load_circuit
from bloqade.cirq_utils.emit import emit_circuit

@squin.kernel
def preparemagicstate(qubit):
    squin.i(qubit)

@squin.kernel
def injection():
    q=squin.qalloc(7)
    preparemagicstate(q[6])
    squin.ry(-pi / 2, q[0])
    squin.ry(-pi / 2, q[1])
    squin.ry(-pi / 2, q[2])
    squin.ry(-pi / 2, q[3])
    squin.ry(-pi / 2, q[4])
    squin.ry(-pi / 2, q[5])
    squin.cz(q[1], q[2])
    squin.cz(q[3], q[4])
    squin.cz(q[5], q[6])
    squin.ry(pi/2,q[6])
    squin.cz(q[0],q[3])
    squin.cz(q[2],q[5])
    squin.cz(q[4],q[6])
    squin.ry(pi/2,q[2])
    squin.ry(pi/2,q[3])
    squin.ry(pi/2,q[4])
    squin.ry(pi/2,q[5])
    squin.ry(pi/2,q[6])
    squin.cz(q[0],q[1])
    squin.cz(q[2],q[3])
    squin.cz(q[4],q[5])
    squin.ry(pi/2,q[1])
    squin.ry(pi/2,q[2])
    squin.ry(pi/2,q[4])

    
# Pass the kernel function (injection), not injection() — calling it would run the interpreter and fail.
cirq_diagram = emit_circuit(injection)
squin_for_diagram = load_circuit(cirq_diagram)
tsim_circ = bloqade.tsim.Circuit(squin_for_diagram)
fig = tsim_circ.diagram(height=400)
# Save diagram as PNG (and optionally SVG). Diagram stores SVG in _svg (may be wrapped in div).
import re

html_or_svg = getattr(fig, "_svg", str(fig))
# Extract inner <svg>...</svg> for conversion (wrap_svg adds a div around the SVG)
svg_match = re.search(r"<svg[\s\S]*?</svg>", html_or_svg)
if svg_match:
    svg_str = svg_match.group(0)
    try:
        import cairosvg
        cairosvg.svg2png(bytestring=svg_str.encode("utf-8"), write_to="msd_diagram.png")
        print("Diagram saved to msd_diagram.png")
    except ImportError:
        with open("msd_diagram.svg", "w", encoding="utf-8") as f:
            f.write(svg_str)
        print("PNG requires cairosvg. Saved msd_diagram.svg instead (pip install cairosvg for PNG).")
    except Exception as e:
        with open("msd_diagram.svg", "w", encoding="utf-8") as f:
            f.write(svg_str)
        print("Saved msd_diagram.svg (PNG failed: %s)." % e)
else:
    print("Diagram created (type: %s). Open in a notebook to display." % type(fig).__name__)