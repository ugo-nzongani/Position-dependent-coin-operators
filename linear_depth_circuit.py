from qiskit import *
from qiskit.tools.visualization import *
from qiskit.extensions import *
from qiskit import QuantumCircuit, transpile
from qiskit.providers.basicaer import QasmSimulatorPy
from qiskit.providers.aer import QasmSimulator
from shift_circuit import *
import qiskit.quantum_info as qi
import numpy as np
import random

"""Linear-depth position-dependent coin operators circuit implementation"""

def l(m):
    """
    Parameters
    ----------
    m : int
        An index used in the computation of Q10

    Returns
    -------
    int
        The index provided by Eq. (C16) in the paper
    """
    if m == 0:
        return 1
    else:
        return 2**(m-1) - 1 + l(m-1)
    
def q10(n):
    """
    Parameters
    ----------
    n : int
        The number of qubits encoding the position

    Returns
    -------
    qiskit.circuit.quantumcircuit.QuantumCircuit
        Quantum circuit Q10 used to build Q1
    """
    N = 2**n
    # Position register
    b = QuantumRegister(n, name= 'b' )
    # Coins register, s[0] is the principal coin
    s = QuantumRegister(N, name= 's' )
    # Ancillary position register
    b_aux = QuantumRegister(N , name= "b'" )
    qc = QuantumCircuit(b, s, b_aux)
    # Adding the quantum gates
    for i in range(n-2 +1):
        # Q_{10}^i
        #qc.barrier()
        for m in range(i+1,n-1+1):
            # J_{m}^i
            if i == 0:
                qc.cnot(b[m],s[l(m)])
            else:
                sm = 0
                for u in range(1,i+1):
                    sm += 2**(u-1)
                qc.cnot(b[m],s[l(m)+sm])
                for l_prime in range(2**i -2+1):
                    qc.cnot(s[l(m)+l_prime],s[l(m)+l_prime+2**i])   
    return qc

def q11(n):
    """
    Parameters
    ----------
    n : int
        The number of qubits encoding the position

    Returns
    -------
    qiskit.circuit.quantumcircuit.QuantumCircuit
        Quantum circuit Q11 used to build Q1
    """
    N = 2**n
    # Position register
    b = QuantumRegister(n, name= 'b' )
    # Coins register, s[0] is the principal coin
    s = QuantumRegister(N, name= 's' )
    # Ancillary position register
    b_aux = QuantumRegister(N , name= "b'" )
    qc = QuantumCircuit(b, s, b_aux)
    # Adding the quantum gates
    #qc.barrier()
    qc.x(b_aux[0])
    for j in range(n-1 +1):
        #qc.barrier()
        qc.cswap(b[j],b_aux[0],b_aux[2**j])
        for k in range(1,2**j -1 +1):
            qc.cswap(s[j+k-1],b_aux[k],b_aux[k+2**j])
    #qc.barrier()     
    return qc

def q2(n):
    """
    Parameters
    ----------
    n : int
        The number of qubits encoding the position

    Returns
    -------
    qiskit.circuit.quantumcircuit.QuantumCircuit
        Quantum circuit Q2
    """
    N = 2**n
    # Position register
    b = QuantumRegister(n, name= 'b' )
    # Coins register, s[0] is the principal coin
    s = QuantumRegister(N, name= 's' )
    # Ancillary position register
    b_aux = QuantumRegister(N , name= "b'" )
    qc = QuantumCircuit(b, s, b_aux)
    # Adding the quantum gates
    for j in range(n-2 +1):
        #qc.barrier()
        for k in range(2**(n-j-1)-2 +1):
            qc.cnot(b_aux[int(2**n -1-2**(j+1)*(1/2 +k))], b_aux[2**n -1-k*2**(j+1)])
    for j in range(n-1+1):
        #qc.barrier()
        for k in range(2**j -1+1):
            qc.cswap(b_aux[(k+1)*2**(n-j)-1], s[k*2**(n-j)], s[int(2**(n-j)*(1/2 +k))])
        if j != n-1:
            #qc.barrier()
            for l in range(2**(j+1)-2+1):
                qc.cnot(b_aux[int(2**n -1-2**(n-j-1)*(1/2 +l))], b_aux[2**n -1- l*2**(n-j-1)])   
    return qc

def q0(n, angles):
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
        Quantum circuit Q0
    """
    # Defining the circuit
    N = 2**n
    # Position register
    b = QuantumRegister(n, name= 'b' )
    # Coins register, s[0] is the principal coin
    s = QuantumRegister(N, name= 's' )
    # Ancillary position register
    b_aux = QuantumRegister(N , name= "b'" )
    qc = QuantumCircuit(b, s, b_aux)
    # Adding the quantum gates
    #qc.barrier()
    for k in range(N):
        theta = angles[k][0]
        phi = angles[k][1]
        lam = angles[k][2]
        gamma = angles[k][3]
        #qc.cu(theta, phi, lam, gamma, b_aux[k], s[k], label="C"+str(k))
        
        array = np.exp(1j*gamma) * np.array([
            [np.cos(theta/2), -np.exp(1j*lam)*np.sin(theta/2)],
            [np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(phi+lam))*np.cos(theta/2)]], dtype=np.complex128)
        gate = UnitaryGate(array, label="C"+str(k))
        qc.append(gate.control(1), [b_aux[k], s[k]])
    #qc.barrier()
    return qc

def build_linear_depth_circuit(n,angles):
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
        Quantum circuit implementing the position-dependent coin operator
    """
    N = 2**n
    # Position register
    b = QuantumRegister(n, name= 'b' )
    # Coins register, s[0] is the principal coin
    s = QuantumRegister(N, name= 's' )
    # Ancillary position register
    b_aux = QuantumRegister(N , name= "b'" )
    qc = QuantumCircuit(b, s, b_aux)
    # All qubits
    all_qubits = [i for i in b] + [i for i in s] + [i for i in b_aux]
    # Adding the Q gates
    # Coin operators
    # Q1
    qc = qc.compose(q10(n),all_qubits)
    qc = qc.compose(q11(n),all_qubits)
    qc = qc.compose(q10(n).inverse(),all_qubits)
    # Q2
    qc = qc.compose(q2(n),all_qubits)
    # Q0
    qc = qc.compose(q0(n, angles),all_qubits)
    # Q2_dagger
    qc = qc.compose(q2(n).inverse(),all_qubits)
    # Q1_dagger
    qc = qc.compose(q10(n),all_qubits)
    qc = qc.compose(q11(n).inverse(),all_qubits)
    qc = qc.compose(q10(n).inverse(),all_qubits)
    return qc

def build_linear_depth_circuit_shift(n,angles,qft):
    """
    Parameters
    ----------
    n : int
        The number of qubits encoding the position
    angles : numpy.ndarray
        Array of size 2**n which contains the angles used to parameterize the coin operators.
        angles[k] = [theta, phi, lam] contains the angles used to parameterize the coin operator 
        applied to the position k
    qft : bool
        True if the shift operator is implemented using the QFT

    Returns
    -------
    qiskit.circuit.quantumcircuit.QuantumCircuit
        Quantum circuit implementing the position-dependent coin operator followed by the shift operator
    """
    N = 2**n
    # Position register
    b = QuantumRegister(n, name= 'b' )
    # Coins register, s[0] is the principal coin
    s = QuantumRegister(N, name= 's' )
    # Ancillary position register
    b_aux = QuantumRegister(N , name= "b'" )
    qc = QuantumCircuit(b, s, b_aux)
    # All qubits
    all_qubits = [i for i in b] + [i for i in s] + [i for i in b_aux]
    # Qubits used for the walk
    walk_qubits = [i for i in b] + [s[0]]
    # Adding the Q gates
    # Coin operators
    qc = qc.compose(build_linear_depth_circuit(n,angles),all_qubits)
    # Shift operator
    if qft:
        qc = qc.compose(qft_shift(n),walk_qubits)
    else:
        qc = qc.compose(shift(n),walk_qubits)
    return qc

def quantum_walk_linear_depth_circuit(n,angles,n_step,qft):
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
    qft : bool
        True if the shift operator is implemented using the QFT

    Returns
    -------
    qiskit.circuit.quantumcircuit.QuantumCircuit
        Quantum circuit implementing the linear-depth position-dependent coin operator followed by the shift operator
        for n_step steps
    """
    N = 2**n
    # Position register
    b = QuantumRegister(n, name= 'b' )
    # Coins register, s[0] is the principal coin
    s = QuantumRegister(N, name= 's' )
    # Ancillary position register
    b_aux = QuantumRegister(N , name= "b'" )
    # Measurement register
    c = ClassicalRegister(n,name="c")
    qc = QuantumCircuit(b, s, b_aux,c)
    # All qubits
    all_qubits = [i for i in b] + [i for i in s] + [i for i in b_aux]
    quantum_circuit = build_linear_depth_circuit_shift(n,angles,qft)
    for i in range(n_step):
        qc = qc.compose(quantum_circuit, all_qubits)
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
        angles[k] = [theta, phi, lam, gamma] contains the angles used to parameterize the coin operator 
        applied to the position k
    """
    N = 2**n
    angles = np.zeros((N, 4))
    for k in range(N):
        theta = random.uniform(0, np.pi)
        phi = random.uniform(-np.pi, np.pi)
        lam = random.uniform(-np.pi, np.pi)
        gamma = random.uniform(0, np.pi)
        angles[k][0] = theta
        angles[k][1] = phi
        angles[k][2] = lam
        angles[k][3] = gamma
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
