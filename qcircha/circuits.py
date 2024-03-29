# This code is part of qcircha.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

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
import numpy as np

__all__ = ['general_qnn', 'circuit12', 'circuit15',
           'circuit9',  'circuit10', 'circuit1',
           'identity', 'mps_circ']

class SU4Gate(QuantumCircuit):
    def __init__(self, name: str = "SU4Gate", param_prefix: str = "x"):
        qc = QuantumCircuit(2, name=name)
        parameters = ParameterVector(name=param_prefix, length=15)

        # Add single-qubit gates
        qc.rz(parameters[0], 0)
        qc.ry(parameters[2], 0)
        qc.rz(parameters[4], 0)

        qc.rz(parameters[1], 1)
        qc.ry(parameters[3], 1)
        qc.rz(parameters[5], 1)

        # Add two-qubit gates
        qc.rxx(parameters[6], 0, 1)
        qc.ryy(parameters[7], 0, 1)
        qc.rzz(parameters[8], 0, 1)

        # Add single-qubit gates
        qc.rz(parameters[9], 0)
        qc.ry(parameters[11], 0)
        qc.rz(parameters[13], 0)

        qc.rz(parameters[10], 1)
        qc.ry(parameters[12], 1)
        qc.rz(parameters[14], 1)

        super().__init__(2, name=name)
        self.append(qc.to_gate(), self.qubits)


def general_qnn(num_reps, feature_map, var_ansatz, alternate=False, barrier=True):
    """
    Creates a general Quantum Neural Network with reuploading given a feature map and a variational (trainable) block.

    Parameters
    ----------
    num_reps : int
        Number of repetitions of the blocks
    feature_map : :py:class:`QuantumCircuit`
        Feature map of the parametrized circuit
    var_ansatz : :py:class:`QuantumCircuit`
        Variational ansatz of the parametrized circuit
    alternate : bool, optional
        If the feature map and the variational ansatz should be alternated in the
        disposition (True) or if first apply ALL the feature map repetitions and
        then all the ansatz repetitions (False). Default to True.
    barrier : bool, optional
        If True, apply barriers after each block, by default True

    Returns
    -------
    :py:class:`QuantumCircuit`
        Parametrized quantum circuit forming the quantum neural network
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

def circuit12(num_qubits, num_reps=1, piramidal=True, barrier=False):
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
        If you should use the piramidal architecture. If False, only
        the first two layers are used, and repeated following the num_reps
        parameter, by default True
    barrier : bool, optional
        If True, insert a barrier after each repetition. Default to False.

    Returns
    -------
    circ : :py:class:`QuantumCircuit`
        The parametric quantum circuit
    """

    circ = QuantumCircuit(num_qubits, name = "circuit12")

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

def circuit15(num_qubits, num_reps=1, barrier=False):
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
    circ : :py:class:`QuantumCircuit`
        The parametric quantum circuit
    """

    circ = QuantumCircuit(num_qubits, name = "circuit15")

    num_params = 2*num_qubits*num_reps
    params = ParameterVector('θ', length=num_params)
    param_idx = 0

    for rep in range(num_reps):
        for ii in range(num_qubits):
            circ.ry(params[param_idx], ii)
            param_idx += 1
        circ.cx(num_qubits-1,0)
        for ii in range(num_qubits-1, 0, -1):
            circ.cx(ii-1, ii)
        for ii in range(num_qubits):
            circ.ry(params[param_idx], ii)
            param_idx += 1

        circ.cx(num_qubits-1, num_qubits-2)
        circ.cx(0, num_qubits-1)
        for ii in range(0, num_qubits-2):
            circ.cx(ii+1, ii)

        if barrier:
            circ.barrier()

    metadata = dict({"entanglement_map": "circular_ish"})
    circ.metadata = metadata

    return circ

def circuit10(num_qubits, num_reps=1, barrier=False):
    """
    Circuit 10 from [1].
    TODO: describe better circuit 10

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
    circ : :py:class:`QuantumCircuit`
        The parametric quantum circuit
    """

    circ = QuantumCircuit(num_qubits, name="circuit10")

    num_params = 2 * num_qubits * num_reps
    params = ParameterVector('θ', length=num_params)

    param_idx = 0
    for rep in range(num_reps):
        for ii in range(num_qubits):
            circ.ry(params[param_idx], ii)
            param_idx += 1

        for ii in range(num_qubits-1):
            circ.cx(ii, ii+1)
        circ.cx(0, num_qubits-1)

        for ii in range(num_qubits):
            circ.ry(params[param_idx], ii)
            param_idx += 1

        if barrier:
            circ.barrier()

    metadata = dict({"entanglement_map": "ring"})
    circ.metadata = metadata

    return circ


def circuit1(num_qubits, num_reps=1, barrier=False):
    """
    Create a dummy/easy qc without entanglement, corresponding to circuit 1 in [1].
    TODO: better describe the circuit

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
    circ : :py:class:`QuantumCircuit`
        The parametric quantum circuit
    """

    circ = QuantumCircuit(num_qubits, name = "circuit1")

    num_params = 2 * num_qubits * num_reps
    params = ParameterVector('θ', length=num_params)

    param_idx = 0
    for rep in range(num_reps):
        for ii in range(num_qubits):
            circ.rx(params[param_idx], ii)
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
    TODO: describe better circuit 9

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
    circ : :py:class:`QuantumCircuit`
        The parametric quantum circuit
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

def identity(num_qubits):
    """
    Identity circuit, does nothing.

    Parameters
    ----------
    num_qubits : int
        Total number of qubits in the system

    Returns
    -------
    circ : :py:class:`QuantumCircuit`
        The Identity quantum circuit
    """

    circ = QuantumCircuit(num_qubits, name="identity", metadata={'entanglement_map': 'None'})
    return circ

def mps_circ(num_qubits):
    bond_dimensions = [2**ii for ii in range(1, int(np.ceil(num_qubits/2)+1))]
    if num_qubits%2 == 0:
        bond_dimensions += bond_dimensions[:-1][::-1]
    else:
        bond_dimensions += bond_dimensions[::-1]
    qc = QuantumCircuit(len(bond_dimensions)+1)
    skip = None
    ii = 0
    for idx, bd in enumerate(bond_dimensions):
        num_qubs = int(np.ceil(np.log2(bd)))

        for nq1 in range(1, num_qubs+1):
            if skip is not None:
                if skip[nq1-1]:
                    continue

            gate_idx = nq1-1
            for nq in range(nq1, 0, -1):
                qc.append(SU4Gate( param_prefix=f"θ{ii}"), (idx+nq, idx+nq-1) )
                if nq - 1 == np.ceil(nq1/2) and nq1%2 == 0:
                    gate_idx = gate_idx
                elif nq-1 > nq1/2:
                    gate_idx -= 2
                else:
                    gate_idx += 2
                ii += 1

        if idx < len(bond_dimensions)-1:
            if bd > bond_dimensions[idx+1] or bd >4:
                skip = [True if nq>1 else False for nq in range(num_qubs)][::-1]
                skip.append(False)
            else:
                skip = None
    metadata = dict({"entanglement_map": "linear"})
    qc.metadata = metadata

    return qc