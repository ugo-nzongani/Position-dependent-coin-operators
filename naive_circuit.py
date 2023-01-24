import random
from qiskit import *
from qiskit.extensions import *
from qiskit.tools.visualization import *
import numpy as np
from qiskit.circuit.library import UGate
from shift_circuit import *
from qiskit.providers.basicaer import QasmSimulatorPy
from qiskit.providers.aer import QasmSimulator
from qiskit import QuantumCircuit, transpile

"""Naive position-dependent coin operators circuit implementation"""

def build_naive_circuit(n,angles):
    """
    Parameters
    ----------
    n : int
        The number of qubits encoding the position
    angles : numpy.ndarray
        Array of size 2**n which contains the angles used to parameterize the coin operators.
        angles[k] = [theta, phi, lam] contains the angles used to parameterize the coin operator 
        applied to the position k

    Returns
    -------
    qiskit.circuit.quantumcircuit.QuantumCircuit
        Quantum circuit implementing the naive position-dependent coin operator followed by the shift operator
        for n_step steps
    """
    N = 2**n
    # Position register
    b = QuantumRegister(n, name= 'b' )
    # Coins register, s[0] is the principal coin
    s = QuantumRegister(1, name= 's' )
    qc = QuantumCircuit(b, s)
    qubits = [i for i in b] + [s[0]]
    for i in range(N):
        for j in range(n):
            if i % 2**(j) == 0:
                qc.x(b[j])
        theta = angles[i][0]
        phi = angles[i][1]
        lam = angles[i][2]
        u = UGate(theta, phi, lam, label='C'+str(i))
        qc.append(u.control(n),qubits)
        #qc.barrier()
    qc = qc.compose(shift(n),qubits)
    return qc

def quantum_walk_naive_circuit(n,angles,n_step):
    """
    Parameters
    ----------
    n : int
        The number of qubits encoding the position
    angles : numpy.ndarray
        Array of size 2**n which contains the angles used to parameterize the coin operators.
        angles[k] = [theta, phi, lam] contains the angles used to parameterize the coin operator 
        applied to the position k
    n_step : int
        The number of steps the walker must take

    Returns
    -------
    qiskit.circuit.quantumcircuit.QuantumCircuit
        Quantum circuit implementing the naive position-dependent coin operator followed by the shift operator
        for n_step steps
    """
    N = 2**n
    # Position register
    b = QuantumRegister(n, name= 'b' )
    # Coins register, s[0] is the principal coin
    s = QuantumRegister(1, name= 's' )
    # Measurement register
    c = ClassicalRegister(n,name="c")
    qc = QuantumCircuit(b, s,c)
    qubits = [i for i in b] + [s[0]]
    quantum_circuit = build_naive_circuit(n,angles)
    for i in range(n_step):
        qc = qc.compose(quantum_circuit, qubits)
    # Measurement of the position register
    qc.measure(b,c)
    return qc

def random_angles(n):
    """
    Parameters
    ----------
    n : int
        The number of qubits encoding the position

    Returns
    -------
    numpy.ndarray
        Array of size 2**n which contains the angles used to parameterize the coin operators.
        angles[k] = [theta, phi, lam] contains the angles used to parameterize the coin operator 
        applied to the position k
    """
    N = 2**n
    angles = np.zeros((N, 3))
    for k in range(N):
        theta = random.uniform(0, np.pi)
        phi = random.uniform(-np.pi, np.pi)
        lam = random.uniform(-np.pi, np.pi)
        angles[k][0] = theta
        angles[k][1] = phi
        angles[k][2] = lam
    return angles

def simulate_circuit(qc,n_shot):
    """
    Parameters
    ----------
    qc : qiskit.circuit.quantumcircuit.QuantumCircuit
        Quantum circuit to simulate
    n_shot : int
        Number of times we want to simulate the execution of the circuit

    Returns
    -------
    qiskit.result.counts.Counts
        A qiskit dictionary containing the results of the measurements
    """
    simulator = QasmSimulator()
    compiled_circuit = transpile(qc, simulator)
    job = simulator.run(compiled_circuit, shots=n_shot)
    result = job.result()
    counts = result.get_counts(qc)
    return counts
