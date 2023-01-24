from qiskit import *

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
        Quantum circuit implementing the shift operator
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
