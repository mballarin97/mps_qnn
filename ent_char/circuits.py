"""
Circuit architectures not present in the qiskit package
"""

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

def piramidal_circuit(num_qubits, num_reps=1, piramidal=True, barrier=False):
    """Create the piramidal circuit generalization 
        corresponding to the circuit 12 of the paper [ADD REF].
    
    Parameters
    ----------
    num_qubits : int
        Total number of qubits in the system
    num_reps : int, optional
        Number of repetitions. It is particolarly important when 
        piramidal=False. By default 1
    piramidal : bool, optional
        If you schould use the piramidal architecture. If False, only
        the first two layers are used, and repeated following the num_reps
        parameter, by default True
    barrier : bool, optional
        If True, insert a barrier after each repetition. Default to False.

    Returns
    -------
    circ : QuantumCircuit
        The parametric quantum circuit
    """
    circ = QuantumCircuit(num_qubits)

    # Compute the number of parameters
    if piramidal:
        num_params = 0
        for ii in range(num_qubits//2):
            for jj in range(ii, num_qubits-ii, 2):
                num_params += 4
    else:
        num_params = 2*num_qubits+ 2*(num_qubits-2)
    num_params *= num_reps

    params = ParameterVector('θ', length=num_params)
    param_idx = 0
    for rep in range(num_reps):
        for ii in range(num_qubits//2):
            # Apply onlt the first two layers if piramidal=False
            if not piramidal:
                if ii>1:
                    break
            for jj in range(ii, num_qubits-ii, 2):
                circ.ry(params[param_idx], jj)
                param_idx += 1
                circ.rz(params[param_idx], jj)
                param_idx += 1
                circ.ry(params[param_idx], jj+1)
                param_idx += 1
                circ.rz(params[param_idx], jj+1)
                param_idx += 1
                circ.cz(jj, jj+1)

        if barrier:
            circ.barrier()

    return circ

def ring_circ(num_qubits, num_reps=1, barrier=False):
    """Create the circuit NN with periodic conditions at the boundaries, 
        corresponding to the circuit 15 of the paper [ADD REF].
    
    Parameters
    ----------
    num_qubits : int
        Total number of qubits in the system
    num_reps : int, optional
        Number of repetitions. By default 1
    barrier : bool, optional
        If True, insert a barrier after each repetition. Default to False.

    Returns
    -------
    circ : QuantumCircuit
        The parametric quantum circuit
    """
    circ = QuantumCircuit(num_qubits)

    num_params = 2*num_qubits*num_reps
    params = ParameterVector('θ', length=num_params)
    param_idx = 0

    for rep in range(num_reps):
        for ii in range(num_qubits):
            circ.ry(params[param_idx], ii)
            param_idx += 1
        circ.cx(0, num_qubits-1)
        for ii in range(num_qubits-1, 0, -1):
            circ.cx(ii-1, ii)
        for ii in range(num_qubits):
            circ.ry(params[param_idx], ii)
            param_idx += 1
        
        circ.cx(num_qubits-1, num_qubits-2)
        circ.cx(0, num_qubits-1)
        for ii in range(0, num_qubits-1):
            circ.cx(ii+1, ii)

        if barrier:
            circ.barrier

    return circ