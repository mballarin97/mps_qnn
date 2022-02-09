"""
Circuit architectures not present in the qiskit package.
Each function returns a Variational Circuit and also a string of its name, for logging reasons.

When adding new circuits, do it in the same format of the other, by:
1. Providing an understandable name to the circuit;
2. Add entanglement map in the metadata information;
3. If other data is needed, then update the general_qnn function to include that data; 

Circuit numbers from Fig. 2 of [1].
Abbass QNN from [2].

Refs:
[1] Expressibility and Entangling Capability of PQCs for Hybrid Quantum-Classical Algorithms, Kim et al. (2019).
[2] The power of QNN, Abbass et al. (2020).

"""

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap, TwoLocal


def general_qnn(num_reps, alternate=False, feature_map=None, var_ansatz=None, barrier=True):
    """
    Creates a general Quantum Neural Network with reuploading given a feature map and a variational (trainable) block.  
    """

    if feature_map.num_qubits != var_ansatz.num_qubits:
        raise TypeError(f"Feature map and variational ansatz have a different number of qubits!")

    param_names = ['θ'+str(i) for i in range(num_reps)]
    param_per_rep = len(var_ansatz.parameters)

    params_list = [ParameterVector(param_names[i], length=param_per_rep) for i in range(num_reps)]

    input_list = ParameterVector('x', length=len(feature_map.parameters))
    feature_map = feature_map.assign_parameters(input_list)

    num_qubits = feature_map.num_qubits
    qc = QuantumCircuit(num_qubits, name="QNN", metadata = {"none": "none"})

    if alternate == True:
        for i in range(num_reps):
            # Feature map first
            qc = qc.compose(feature_map)
            if barrier:
                qc.barrier()

            # Variational ansatz (renaming params to avoid naming conflicts)
            var_ansatz = var_ansatz.assign_parameters(params_list[i])
            qc = qc.compose(var_ansatz)
            if barrier:
                qc.barrier()

    elif alternate == False:
        # Feature map firs repeated reps times
        for i in range(num_reps):
            qc = qc.compose(feature_map)
            if barrier:
                qc.barrier()

        # Variational ansatz repeated reps times
        for i in range(num_reps):
            var_ansatz = var_ansatz.assign_parameters(params_list[i])
            qc = qc.compose(var_ansatz)
            if barrier:
                qc.barrier()
    else:
        raise TypeError(f"Structure {alternate} not implemented. ")

    # Extract feature map entanglement map
    try: 
        feature_map.entanglement
    except:
        fmap_entanglement = feature_map.metadata["entanglement_map"]
    else:  
        fmap_entanglement = feature_map.entanglement
    
    # Extract variational ansatz entanglement map
    try:
        var_ansatz.entanglement
    except:
        var_entanglement = var_ansatz.metadata["entanglement_map"]
    else:
        var_entanglement = var_ansatz.entanglement

    data = dict({"num_qubits": num_qubits,
                 "num_reps": num_reps,
                 "alternate": alternate,
                 "params": len(qc.parameters),
                 "fmap": feature_map.name,
                 "fmap_entanglement": fmap_entanglement,
                 "var_ansatz": var_ansatz.name,
                 "var_entanglement": var_entanglement
                })
    qc.metadata = data

    # @TODO: Rename trainable parameters as θ[0],...,θ[p].

    return qc


def Abbas_QNN(num_qubits, reps=2, alternate=True, barrier=False):
    """
    Creates the QNN from Abbas, given by repetition of ZZfeaturemap and TwoLocal vartiaional ansatz.
    """

    param_names = ['θ'+str(i) for i in range(0, reps)]
    feature_map = ZZFeatureMap(num_qubits, reps=1, entanglement='linear', insert_barriers=barrier)

    qc = QuantumCircuit(num_qubits, name = "AbbassQNN")

    if alternate:
        for i in range(reps):
            qc = qc.compose(feature_map)

            # Skip final rotation to ensure the alternate = T/F have same number of params
            var_ansatz = TwoLocal(num_qubits, 'ry', 'cx', 'linear', reps=1,
                                  skip_final_rotation_layer=True, insert_barriers=barrier, parameter_prefix=param_names[i])

            if barrier:
                qc.barrier()

            qc = qc.compose(var_ansatz)
            
            if barrier:
                qc.barrier()

    elif alternate == False:
        qc = qc.compose(ZZFeatureMap(num_qubits, reps=reps,
                        entanglement='linear', insert_barriers=barrier))
        
        if barrier:
            qc.barrier()
        
        # Skip final rotation to ensure the alternate = T/F have same number of params
        qc = qc.compose(TwoLocal(num_qubits, 'ry', 'cx', 'linear', reps=reps, 
                                skip_final_rotation_layer=True, insert_barriers=barrier))
    else:
        raise TypeError(f"Structure {alternate} not implemented. ")

    return qc

def piramidal_circuit(num_qubits, num_reps=1, piramidal=True, barrier=False):
    """
    Create the piramidal circuit generalization corresponding to the circuit 12 of the paper [1].
    
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

    circ = QuantumCircuit(num_qubits, name = "piramidal_circuit")

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

    metadata = dict({"entanglement_map": "linear"})
    circ.metadata = metadata

    return circ

def ring_circ(num_qubits, num_reps=1, barrier=False):
    """
    Create the circuit NN with periodic conditions at the boundaries, corresponding to the circuit 15 of the paper [1].
    
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

    circ = QuantumCircuit(num_qubits, name = "ring_circ")

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
            circ.barrier()

    metadata = dict({"entanglement_map": "circular_ish"})
    circ.metadata = metadata

    return circ


def dummy_circ(num_qubits, num_reps=1, barrier=False):
    """
    Create a dummy/easy qc.
    """

    circ = QuantumCircuit(num_qubits, name="dummy_circ")

    num_params = 2 * num_qubits * num_reps
    params = ParameterVector('θ', length=num_params)
    
    param_idx = 0
    for rep in range(num_reps):
        for ii in range(num_qubits):
            circ.ry(params[param_idx], ii)
            param_idx += 1
            circ.rz(params[param_idx], ii)
            param_idx += 1
        
        if barrier:
            circ.barrier()

    metadata = dict({"entanglement_map": "none"})
    circ.metadata = metadata

    return circ


def circuit9(num_qubits, num_reps=1, barrier=False):
    """
    Circuit 9 from [1].
    """

    circ = QuantumCircuit(num_qubits, name="circuit9")

    num_params = 1 * num_qubits * num_reps
    params = ParameterVector('θ', length=num_params)

    param_idx = 0
    for rep in range(num_reps):
        for ii in range(num_qubits):
            circ.h(ii)
        
        for ii in range(num_qubits-1):
            circ.cz(ii, ii+1)
        
        for ii in range(num_qubits):
            circ.rx(params[param_idx], ii)
            param_idx += 1

        if barrier:
            circ.barrier()

    metadata = dict({"entanglement_map": "linear"})
    circ.metadata = metadata

    return circ
