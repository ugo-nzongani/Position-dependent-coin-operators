import random
import numpy as np
from qiskit.result import Counts

"""Matrix simulation of quantum walk with position-dependent coin operators"""

def shift_operator(n):
    """
    Parameters
    ----------
    n : int
        The number of qubits encoding the position

    Returns
    -------
    numpy.ndarray
        Shift operator
    """
    N = 2**n
    s0 = np.zeros((N,N))
    s1 = np.zeros((N,N))
    for i in range(N):
        s0[(i-1) % N][i] = 1
        s1[(i+1) % N][i] = 1
    s = np.kron(s0, [[1,0],[0,0]]) + np.kron(s1, [[0,0],[0,1]])
    return s

def unitary(theta,phi,lam,gamma):
    """
    Parameters
    ----------
    theta : float
        Euler angle theta
    phi : float
        Euler angle phi
    lam : float
        Euler angle lambda
    gamma : float
        Global phase, exp(i*gamma)

    Returns
    -------
    numpy.ndarray
        Unitary parametrized with 3 Euler angles and a global phase
    """
    u = np.zeros((2,2), dtype=np.complex128)
    u[0][0] = np.cos(theta/2)
    u[0][1] = -np.exp(1j*lam) * np.sin(theta/2)
    u[1][0] = np.exp(1j*phi) * np.sin(theta/2)
    u[1][1] = np.exp(1j*(phi+lam)) * np.cos(theta/2)
    u *= np.exp(1j*gamma)
    return u

def position_dependent_coin_operators(n,angles):
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
    numpy.ndarray
        Position-dependent coin operator
    """
    N = 2**n
    coin_temp = []
    for i in range(N):
        coin_operator = unitary(angles[i][0],angles[i][1],angles[i][2],angles[i][3])
        c = np.zeros((N,N), dtype=np.complex128)
        c[i][i] = 1
        c = np.kron(c,coin_operator)
        coin_temp.append(c)
    c = np.zeros((2*N,2*N), dtype=np.complex128)
    for i in coin_temp:
        c += i
    return c

def quantum_walk_simulation(n,v,angles,n_step):
    """
    Parameters
    ----------
    n : int
        The number of qubits encoding the position
    v : numpy.ndarray
        Quantum state of the walker position and the coin
    angles : numpy.ndarray
        Array of size 2**n which contains the angles used to parameterize the coin operators.
        angles[k] = [theta, phi, lam] contains the angles used to parameterize the coin operator 
        applied to the position k
    n_step : int
        The number of steps the walker must take

    Returns
    -------
    numpy.ndarray
        Quantum state of the walker position and the coin after n_step steps
    """
    s = shift_operator(n)
    c = position_dependent_coin_operators(n,angles)
    w = np.matmul(s,c)
    full_w = np.linalg.matrix_power(w, n_step)
    return np.matmul(full_w, v)

def position_state(n,init={0:1}):
    """
    Parameters
    ----------
    n : int
        The number of qubits encoding the position
    init : dict
        Dictionary where the keys represent the basis position states 
        and the values are the associated probability amplitudes.
        The only basis states present are those with a non-zero probability amplitude

    Returns
    -------
    numpy.ndarray
        Normalized quantum position state of the walker
    """
    N = 2**n
    v = np.zeros((N), dtype=np.complex128)
    for i in init.keys():
        v[i] = init[i]
    # Normalize
    norm = 0
    for i in v:
        norm += np.absolute(i)**2
    if(norm != 0):
        v /= np.sqrt(norm)
    return v

def coin_state(init={0:1}):
    """
    Parameters
    ----------
    init : dict
        Dictionary where the keys represent the basis coin states 
        and the values are the associated probability amplitudes.
        The only basis states present are those with a non-zero probability amplitude

    Returns
    -------
    numpy.ndarray
        Normalized quantum coin state
    """
    v = np.zeros((2), dtype=np.complex128)
    for i in init.keys():
        v[i] = init[i]
    # Normalize
    norm = 0
    for i in v:
        norm += np.absolute(i)**2
    if(norm != 0):
        v /= np.sqrt(norm)
    return v

def quantum_state(position, coin):
    """
    Parameters
    ----------
    position : numpy.ndarray
        Position quantum state
    coin : numpy.ndarray
        Coin quantum state

    Returns
    -------
    numpy.ndarray
        Quantum state representing the tensor product between position and coin state
    """
    return np.kron(position,coin)

def results(n,v):
    """
    Parameters
    ----------
    n : int
        The number of qubits encoding the position
    v : numpy.ndarray
        Coin quantum state

    Returns
    -------
    dict
        Dictionary where the keys are the different basis states
        and the values are the associated measurement probabilities.
        The only basis states present are those with a non-zero probability
    """
    r = {}
    dim = v.shape[0]
    for i in range(int(dim/2)):
        coef_position_i_coin0 = v[2*i]
        coef_position_i_coin1 = v[2*i + 1]
        if coef_position_i_coin0 != 0 or coef_position_i_coin1 != 0:
            binary = bin(i)[2:].zfill(n)
            probability = np.absolute(coef_position_i_coin0)**2 + np.absolute(coef_position_i_coin1)**2
            r[str(binary)] = probability
    return r

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