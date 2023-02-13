from qiskit import *
from qiskit.circuit.library import QFT
from numpy import pi

"""Fujiwara's shift operator circuit implementation"""

def shift_gates(n):
    """
    Parameters
    ----------
    n : int
        The number of qubits encoding the position

    Returns
    -------
    (qiskit.circuit.gate.Gate,qiskit.circuit.gate.Gate)
        Quantum gates Z+ and Z- used to move the walker
    """
    # Position register
    b = QuantumRegister(n, name="b")
    # Coin register
    s = QuantumRegister(1, name="s")
    qc = QuantumCircuit(b, s)
    for i in range(n):
        ctrl = [s[0]] + b[:n-i-1]
        qc.mct(ctrl, [b[n-i-1]])
    z_up = qc.to_gate(label="Z+")
    z_down = qc.inverse().to_gate(label="Z-")
    return z_up, z_down

def shift(n):
    """
    Parameters
    ----------
    n : int
        The number of qubits encoding the position

    Returns
    -------
    qiskit.circuit.quantumcircuit.QuantumCircuit
        Quantum circuit implementing the shift operator introduced by Fujiwara et al. (10.1103/PhysRevA.72.032329)
    """
    # Position register
    b = QuantumRegister(n, name="b")
    # Coin register
    s = QuantumRegister(1, name="s")
    qc = QuantumCircuit(b, s)
    qubits = [i for i in b] + [s[0]]
    z_up, z_down = shift_gates(n)
    qc.append(z_up, qubits)
    #qc.barrier()
    qc.x(s[0])
    #qc.barrier()
    qc.append(z_down, qubits)
    #qc.barrier()
    qc.x(s[0])
    #qc.barrier()
    return qc


"""QFT's shift operator circuit implementation from Asif Shakeel's paper"""

def qft_shift(n):
    """
    Parameters
    ----------
    n : int
        The number of qubits encoding the position

    Returns
    -------
    qiskit.circuit.quantumcircuit.QuantumCircuit
        Quantum circuit implementing the shift operator using the QFT introduced by Asif Shakeel (10.1007/s11128-020-02834-y)
    """
    # Position register
    b = QuantumRegister(n, name="b")
    # Coin register
    s = QuantumRegister(1, name="s")
    # Create a quantum circuit using the quantum and classical registers
    qc = QuantumCircuit(b, s)

    position_qubits = [i for i in b]
    #position_qubits.reverse()
    
    qft = QFT(n, do_swaps=False)
    # We put a NOT gate because in his paper the walker goes on the left when the coin is |1>
    # But in our implementation it's actually when its value is |0>
    qc.x(s[0])
    for i in range(n):
        qc.cx(s[0], b[n-i-1])
    qc.append(qft, position_qubits)
    for i in range(n):
        qc.p(2*pi/2**(n-i), n-i-1)
    qc.append(qft.inverse(), position_qubits)
    for i in range(n):
        qc.cx(s[0], b[i])
    qc.x(s[0])
    return qc
