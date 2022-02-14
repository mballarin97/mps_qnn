
import numpy as np
from qiskit.quantum_info.states.utils import partial_trace
from qiskit.quantum_info.states.measures import entropy

def entanglement_entropy(statevector, idx_to_trace=None):
    """
    Entanglement entropy of subsystem of a pure state.
    Given a statevector (i.e. pure state), builds the density matrix,
    and traces out some systems.
    Then eveluates Von Neumann entropy using Qiskit's implementation.
    Be consistent with the base of the logarithm.

    Parameters
    ----------
    statevector : array-like
        Statevector of the system
    idx_to_trace : array-like, optional
        Indexes to trace away. By default None.
    
    Returns
    -------
    float
        Entanglement entropy of the reduced density matrix obtained from statevector
        by tracing away the indexes selected
    """
    # Construct density matrix
    rho = np.outer(statevector, np.conjugate(statevector))
    # Trace out part of the system (indicated by idx_to_trace, following Qiskit indexing)
    partial_rho = partial_trace(rho, idx_to_trace)
    ent_entropy = entropy(partial_rho, base = np.e)

    return ent_entropy

def entanglement_bond(statevector):
    """
    Evaluates the entanglement entropies for cuts in a 1-d quantum system (as in MPS), given a pure state 
    statevector of num_qubits qubits, thus with length 2**num_qubits.
    Must be a valid statevector: amplitudes summing to one.

    Parameters
    ----------
    statevector : array-like
        Statevector of the system
    
    Returns
    -------
    array-like of floats
        Entanglement entropy along all the bipartitions of the system
    """
    sum_amplitudes = np.sum(np.vdot(statevector, statevector) )
    if not np.isclose(sum_amplitudes, 1):
        raise ValueError(f'Amplitudes of the statevector must sum up to 1, not to {sum_amplitudes}')

    num_qubits = int(np.log2(len(statevector)))
    
    # Cut along the first half of the bond
    res1 = [entanglement_entropy(statevector, idx_to_trace=range(i+1, num_qubits)) for i in range(int(num_qubits/2))]
    
    # Cut along the second half
    res2 = [entanglement_entropy(statevector, idx_to_trace=range(0, i)) for i in range(1+int(num_qubits/2), num_qubits)]
    res = res1 + res2

    return np.array(res)