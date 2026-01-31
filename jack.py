## implement fig A4
import numpy as np
from bloqade import squin
from math import pi
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

    ###########
    @squin.kernel
    def encode_a4():
        q = squin.qalloc(14)
        squin.h(q[8])
        squin.h(q[12])
        squin.h(q[13])

        ##
        squin.cx(q[8], q[4])
        squin.cx(q[6], q[10])
        squin.cx(q[5], q[10])
        squin.cx(q[8],q[10])
        squin.cx(q[8], q[0])
        squin.cx(q[4], q[9])
        squin.cx(q[1], q[10])
        squin.cx(q[8], q[2])