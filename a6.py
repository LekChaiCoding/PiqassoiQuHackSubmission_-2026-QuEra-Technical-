###########
## a6.py ##
###########

from bloqade import squin
from bloqade.types import Qubit
from kirin.dialects import ilist
from bloqade.cirq_utils import load_circuit
from bloqade.cirq_utils.emit import emit_circuit
import bloqade.stim
import numpy as np
from math import pi
from typing import Literal


#flag qbit 11 (index 10)
def a6_circuit():
    q = squin.qalloc(11)

    #|0> qbits 6-11 (index 5-10)
    for i in range(5,11):
        squin.reset(q[i])
    
    squin.h(q[5])
    #cx gates on 6 target 7, then 7 target 8, etc up to 11.
    for i in range(5,10):
        squin.cx(q[i],q[i+1])
    
    squin.cx(q[6], q[10])


    # Load the ghz circuit for qbits 1-5 (index 0-4)

    # cx gates on 1 target 6, 2 t 7, etc up to 5 t 10
    for i in range(0,5):
        squin.cx(q[i],q[i+6])
    
    # Measure qbits 6-10 (index 5-9)
    for i in range(5,10):
        squin.measure(q[i])
    

    
