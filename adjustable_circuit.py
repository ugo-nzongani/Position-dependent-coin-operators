from qiskit import *
from qiskit.tools.visualization import *
from qiskit.extensions import *
from qiskit import QuantumCircuit, transpile
from qiskit.providers.basicaer import QasmSimulatorPy
from qiskit.providers.aer import QasmSimulator
from shift_circuit import *
import numpy as np
import random

"""Adjustable-depth position-dependent coin operators circuit implementation"""

def l(k):
    """
    Parameters
    ----------
    k : int
        An index used in the computation of Q10

    Returns
    -------
    int
        The index provided by Eq. (A5) in the paper
    """
    if k == 0:
        return 1
    else:
        return 2**(k-1) - 1 + l(k-1)

def q10(n,m):
    """
    Parameters
    ----------
    n : int
        The number of qubits encoding the position
    m : int
        The parameter of the circuit

    Returns
    -------
    qiskit.circuit.quantumcircuit.QuantumCircuit
        Quantum circuit Q10 used to build Q1
    """
    M = 2**m
    # Position register
    b = QuantumRegister(n, name= 'b' )
    # Coins register, s[0] is the principal coin
    s = QuantumRegister(M, name= 's' )
    # Ancillary position register
    b_aux = QuantumRegister(M , name= "b'" )
    qc = QuantumCircuit(b, s, b_aux)
    # Adding the quantum gates
    for j in range(m-2 +1):
        #qc.barrier()
        for k in range(j+1,m-1+1):
            if j == 0:
                qc.cnot(b[k],s[l(k)])
            else:
                sm = 0
                for u in range(1,j+1):
                    sm += 2**(u-1)
                qc.cnot(b[k],s[l(k)+sm])
                for l_prime in range(2**j -2+1):
                    qc.cnot(s[l(k)+l_prime],s[l(k)+l_prime+2**j])   
    return qc

def q11(n,m,i,optimized=True):
    """
    Parameters
    ----------
    n : int
        The number of qubits encoding the position
    m : int
        The parameter of the circuit
    i : int
        Iteration index of U_i operator  
    optimized : bool
        True if we optimize the number of X gates for bit flips before the (n-m)-Toffoli gate

    Returns
    -------
    qiskit.circuit.quantumcircuit.QuantumCircuit
        Quantum circuit Q11 used to build Q1
    """
    M = 2**m
    # Position register
    b = QuantumRegister(n, name= 'b' )
    # Coins register, s[0] is the principal coin
    s = QuantumRegister(M, name= 's' )
    # Ancillary position register
    b_aux = QuantumRegister(M , name= "b'" )
    qc = QuantumCircuit(b, s, b_aux)
    # Adding the quantum gates
    #qc.barrier()
    
    ctrl_qubits = [b[k] for k in range(m,n)] # called alpha in the paper
    # Bits flips
    for k in range(m,n):
        # Function g
        if optimized:
            if i % 2**(k-m) == 0:
                qc.x(b[k])
        else:
            if np.floor((i/2**(k-m)) % 2) == 0:
                qc.x(b[k])
    # (n-m)-Toffoli gate
    if m != n :
        qc.mct(ctrl_qubits,b_aux[0])
    else:
        qc.x(b_aux[0])    
    # Controlled-swaps
    for j in range(m-1 +1):
        #qc.barrier()
        qc.cswap(b[j],b_aux[0],b_aux[2**j])
        for k in range(1,2**j -1 +1):
            qc.cswap(s[j+k-1],b_aux[k],b_aux[k+2**j])
    #qc.barrier()     
    return qc

def q11_hat(n,m,i):
    """
    Parameters
    ----------
    n : int
        The number of qubits encoding the position
    m : int
        The parameter of the circuit
    i : int
        Iteration index of U_i operator

    Returns
    -------
    qiskit.circuit.quantumcircuit.QuantumCircuit
        Quantum circuit Q11 used to build Q1
    """
    M = 2**m
    # Position register
    b = QuantumRegister(n, name= 'b' )
    # Coins register, s[0] is the principal coin
    s = QuantumRegister(M, name= 's' )
    # Ancillary position register
    b_aux = QuantumRegister(M , name= "b'" )
    qc = QuantumCircuit(b, s, b_aux)
    # Adding the quantum gates
    #qc.barrier()
    
    ctrl_qubits = [b[k] for k in range(m,n)] # called alpha in the paper
    # (n-m)-Toffoli gate
    if m != n :
        qc.mct(ctrl_qubits,b_aux[0])
    else:
        qc.x(b_aux[0])    
    # Controlled-swaps
    for j in range(m-1 +1):
        #qc.barrier()
        qc.cswap(b[j],b_aux[0],b_aux[2**j])
        for k in range(1,2**j -1 +1):
            qc.cswap(s[j+k-1],b_aux[k],b_aux[k+2**j])
    #qc.barrier()     
    return qc


def q2(n,m):
    """
    Parameters
    ----------
    n : int
        The number of qubits encoding the position
    m : int
        The parameter of the circuit

    Returns
    -------
    qiskit.circuit.quantumcircuit.QuantumCircuit
        Quantum circuit Q2
    """
    M = 2**m
    # Position register
    b = QuantumRegister(n, name= 'b' )
    # Coins register, s[0] is the principal coin
    s = QuantumRegister(M, name= 's' )
    # Ancillary position register
    b_aux = QuantumRegister(M , name= "b'" )
    qc = QuantumCircuit(b, s, b_aux)
    # Adding the quantum gates
    # Q20
    for j in range(m-2 +1):
        #qc.barrier()
        for k in range(2**(m-j-1)-2 +1):
            qc.cnot(b_aux[int(2**m -1-2**(j+1)*(1/2 +k))], b_aux[2**m -1-k*2**(j+1)])
    # Q21
    for j in range(m-1+1):
        #qc.barrier()
        for k in range(2**j -1+1):
            qc.cswap(b_aux[(k+1)*2**(m-j)-1], s[k*2**(m-j)], s[int(2**(m-j)*(1/2 +k))])
        if j != m-1:
            #qc.barrier()
            for l in range(2**(j+1)-2+1):
                qc.cnot(b_aux[int(2**m -1-2**(m-j-1)*(1/2 +l))], b_aux[2**m -1- l*2**(m-j-1)])   
    return qc

def q0(n,m,i,angles):
    """
    Parameters
    ----------
    n : int
        The number of qubits encoding the position
    m : int
        The parameter of the circuit
    i : int
        Iteration index of U_i operator
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
    M = 2**m
    # Position register
    b = QuantumRegister(n, name= 'b' )
    # Coins register, s[0] is the principal coin
    s = QuantumRegister(M, name= 's' )
    # Ancillary position register
    b_aux = QuantumRegister(M , name= "b'" )
    qc = QuantumCircuit(b, s, b_aux)
    # Adding the quantum gates
    #qc.barrier()
    for k in range(M):
        current_k = i*M + k
        theta = angles[current_k][0]
        phi = angles[current_k][1]
        lam = angles[current_k][2]
        gamma = angles[current_k][3]
        qc.cu(theta, phi, lam, gamma, b_aux[k], s[k],label="C"+str(k))
        '''
        array = np.exp(1j*gamma) * np.array([
            [np.cos(theta/2), -np.exp(1j*lam)*np.sin(theta/2)],
            [np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(phi+lam))*np.cos(theta/2)]], dtype=np.complex128)
        gate = UnitaryGate(array, label="C"+str(k))
        qc.append(gate.control(1), [b_aux[k], s[k]])
        '''
    #qc.barrier()
    return qc

def build_u_i(n,m,i,angles,optimized=True):
    """
    Parameters
    ----------
    n : int
        The number of qubits encoding the position
    m : int
        The parameter of the circuit
    i : int
        Iteration index of U_i operator
    angles : numpy.ndarray
        Array of size 2**n which contains the angles used to parameterize the coin operators.
        angles[k] = [theta, phi, lam] contains the angles used to parameterize the coin operator 
        applied to the position k
    optimized : bool
        True if we optimize the number of X gates for bit flips before the (n-m)-Toffoli gate

    Returns
    -------
    qiskit.circuit.quantumcircuit.QuantumCircuit
        Quantum circuit U_i
    """
    M = 2**m
    # Position register
    b = QuantumRegister(n, name= 'b' )
    # Coins register, s[0] is the principal coin
    s = QuantumRegister(M, name= 's' )
    # Ancillary position register
    b_aux = QuantumRegister(M , name= "b'" )
    qc = QuantumCircuit(b, s, b_aux)
    all_qubits = [i for i in b] + [i for i in s] + [i for i in b_aux]
    #qc.barrier()
    # Q1
    qc = qc.compose(q10(n,m),all_qubits)
    qc = qc.compose(q11(n,m,i,optimized),all_qubits)
    qc = qc.compose(q10(n,m).inverse(),all_qubits)
    # Q2
    qc = qc.compose(q2(n,m),all_qubits)
    # Q0
    qc = qc.compose(q0(n,m,i,angles),all_qubits)
    # Q2_dagger
    qc = qc.compose(q2(n,m).inverse(),all_qubits)
    # Q1_dagger
    qc = qc.compose(q10(n,m),all_qubits)
    if optimized:
        qc = qc.compose(q11_hat(n,m,i).inverse(),all_qubits)
    else:
        qc = qc.compose(q11(n,m,i,optimized=False).inverse(),all_qubits)
    qc = qc.compose(q10(n,m).inverse(),all_qubits)
    #qc.barrier()
    return qc
    
def build_adjustable_depth_circuit(n,m,angles,optimized=True):
    """
    Parameters
    ----------
    n : int
        The number of qubits encoding the position
    m : int
        The parameter of the circuit
    angles : numpy.ndarray
        Array of size 2**n which contains the angles used to parameterize the coin operators.
        angles[k] = [theta, phi, lam] contains the angles used to parameterize the coin operator 
        applied to the position k
    optimized : bool
        True if we optimize the number of X gates for bit flips before the (n-m)-Toffoli gate

    Returns
    -------
    qiskit.circuit.quantumcircuit.QuantumCircuit
        Quantum circuit implementing the position-dependent coin operator followed by the shift operator
    """
    M = 2**m
    # Position register
    b = QuantumRegister(n, name= 'b' )
    # Coins register, s[0] is the principal coin
    s = QuantumRegister(M, name= 's' )
    # Ancillary position register
    b_aux = QuantumRegister(M , name= "b'" )
    qc = QuantumCircuit(b, s, b_aux)
    # All qubits
    all_qubits = [i for i in b] + [i for i in s] + [i for i in b_aux]
    for i in range(2**(n-m)):
        qc = qc.compose(build_u_i(n,m,i,angles,optimized),all_qubits)
    return qc

def build_adjustable_depth_circuit_shift(n,m,angles,qft,optimized=True):
    """
    Parameters
    ----------
    n : int
        The number of qubits encoding the position
    m : int
        The parameter of the circuit
    angles : numpy.ndarray
        Array of size 2**n which contains the angles used to parameterize the coin operators.
        angles[k] = [theta, phi, lam] contains the angles used to parameterize the coin operator 
        applied to the position k
    qft : bool
        True if the shift operator is implemented using the QFT
    optimized : bool
        True if we optimize the number of X gates for bit flips before the (n-m)-Toffoli gate

    Returns
    -------
    qiskit.circuit.quantumcircuit.QuantumCircuit
        Quantum circuit implementing the position-dependent coin operator followed by the shift operator
    """
    M = 2**m
    # Position register
    b = QuantumRegister(n, name= 'b' )
    # Coins register, s[0] is the principal coin
    s = QuantumRegister(M, name= 's' )
    # Ancillary position register
    b_aux = QuantumRegister(M, name= "b'" )
    qc = QuantumCircuit(b, s, b_aux)
    # All qubits
    all_qubits = [i for i in b] + [i for i in s] + [i for i in b_aux]
    # Qubits used for the walk
    walk_qubits = [i for i in b] + [s[0]]
    # Adding the Q gates
    # Coin operators
    qc = qc.compose(build_adjustable_depth_circuit(n,m,angles,optimized),all_qubits)
    # Shift operator
    if qft:
        qc = qc.compose(qft_shift(n),walk_qubits)
    else:
        qc = qc.compose(shift(n),walk_qubits)
    return qc

def quantum_walk_adjustable_depth_circuit(n,m,angles,n_step,qft,optimized=True):
    """
    Parameters
    ----------
    n : int
        The number of qubits encoding the position
    m : int
        Parameter of the circuit
    angles : numpy.ndarray
        Array of size 2**n which contains the angles used to parameterize the coin operators.
        angles[k] = [theta, phi, lam] contains the angles used to parameterize the coin operator 
        applied to the position k
    n_step : int
        The number of steps the walker must take
    qft : bool
        True if the shift operator is implemented using the QFT
    optimized : bool
        True if we optimize the number of X gates for bit flips before the (n-m)-Toffoli gate

    Returns
    -------
    qiskit.circuit.quantumcircuit.QuantumCircuit
        Quantum circuit implementing the linear-depth position-dependent coin operator followed by the shift operator
        for n_step steps
    """
    M = 2**m
    # Position register
    b = QuantumRegister(n, name= 'b' )
    # Coins register, s[0] is the principal coin
    s = QuantumRegister(M, name= 's' )
    # Ancillary position register
    b_aux = QuantumRegister(M, name= "b'" )
    # Measurement register
    c = ClassicalRegister(n,name="c")
    qc = QuantumCircuit(b, s, b_aux,c)
    # All qubits
    all_qubits = [i for i in b] + [i for i in s] + [i for i in b_aux]
    quantum_circuit = build_adjustable_depth_circuit_shift(n,m,angles,qft,optimized)
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
        angles[k] = [theta, phi, lam] contains the angles used to parameterize the coin operator 
        applied to the position k
    """
    N = 2**n
    angles = np.zeros((N, 3))
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
