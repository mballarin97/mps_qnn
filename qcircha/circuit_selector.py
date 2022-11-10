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
Functions to create and manage QNNs, in addition to some pre-defined structures.
It is used as an intermediate step to create QNNs using the definitions
in the script `circuit.py`. Circuits for the `ZZFeatureMap` and `TwoLocal`
schemes with all possible entangling topologies are defined.
"""

# Import necessary modules
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, TwoLocal

from qcircha.circuits import *
from qcircha.circuits import circuit_adjm

import numpy as np


def pick_circuit(num_qubits, num_reps, feature_map = 'ZZFeatureMap', 
                 var_ansatz = 'TwoLocal', alternate=True, va_mat_adj=np.zeros((1,1)), va_layer_ry=True, fm_mat_adj=np.zeros((1,1)), fm_layer_ry=True):
    """
    Select a circuit with a feature map and a variational block. Examples below.
    Each block must have reps = 1, and then specify the correct number of repetitions
    is ensured by the :py:func:`general_qnn`.

    Available circuits:

    - 'ZZFeatureMap' : circuit with linear entanglement, used as feature map in the Power of Quantum Neural networks by Abbas et al.
    - 'TwoLocal' : circuit with linear entanglement, used as ansatz in the Power of Quantum Neural networks by Abbas et al.
    - 'Circuit15' : circuit with ring entanglement, defined in (n.15 from Kim et al.)
    - 'Circuit12' : circuit with linear entanglement and a piramidal structure, defined in (n.12 from Kim et al.)
    - 'Circuit1' : circuit without entanglement with two single qubits rotations per qubit (n.1 from Kim et al.)
    - 'Identity' : identity circuit
    - 'Circuit9' : circuit n. 9 from Kim et al.
    - 'Circuit_adjm' : circuit with entanglement map given by the adjacency matrix
    - variations of TwoLocal structures are present.

    Parameters
    ----------
    num_qubits : int
        Number of qubits
    num_reps : int
        Number of repetitions
    feature_map : str or :py:class:`QuantumCircuit`, optional
        Type of feature map. Available options in the description. If a
        :py:class:`QuantumCircuit` it is used instead of the default ones.
        Default to 'ZZFeatureMap'.
    ansatz : str or :py:class:`QuantumCircuit`, optional
        Type of feature map. Available options in the description. If a
        :py:class:`QuantumCircuit` it is used instead of the default ones.
        Default to 'TwoLocal'.
    alternate : bool, optional
        If the feature map and the variational ansatz should be alternated in the
        disposition (True) or if first apply ALL the feature map repetitions and
        then all the ansatz repetitions (False). Default to True.

    Return
    ------
    :py:class:`QuantumCircuit`
        quantum circuit with the correct structure
    """

    feature_map = _select_circ(num_qubits, feature_map, fm_mat_adj, fm_layer_ry)
    var_ansatz = _select_circ(num_qubits, var_ansatz, va_mat_adj, va_layer_ry)

    # Build the PQC
    ansatz = general_qnn(num_reps, feature_map=feature_map,
                         var_ansatz=var_ansatz, alternate=alternate, barrier=True)

    return ansatz


def _select_circ(num_qubits, circ = 'ZZFeatureMap', mat_adj=np.zeros((1,1)), layer_ry=True): 
    #circ = 'ZZFeatureMap' definito di default,ma quando la chiamo gli assegno quella che voglio ovviamente
    """
    Select the circuit based on the possibilities

    Available circuits:
    - 'ZZFeatureMap' : circuit with linear entanglement, used as feature map
        in the Power of Quantum Neural networks by Abbas et al.
    - 'TwoLocal' : circuit with linear entanglement, used as ansatz in
        the Power of Quantum Neural networks by Abbas et al.
    - 'Circuit15' : circuit with ring entanglement, defined in
        (n.15 from Kim et al.)
    - 'Circuit12' : circuit with linear entanglement and a piramidal
        structure, defined in (n.12 from Kim et al.)
    - 'Circuit1' : easy circuit without entanglement (n.1 from Kim et al.)
    - 'circuit9' : circuit n. 9 from Kim et al.

    Parameters
    ----------
    num_qubits : int
        Number of qubits
    circ : str or :py:class:`QuantumCircuit`, optional
        Type of circuit. Available options in the description. If a
        :py:class:`QuantumCircuit` it is used instead of the default ones.
        Default to 'ZZFeatureMap'.

    Return
    ------
    :py:class:`QuantumCircuit`
        Selected quantum circuit
    """

    # If it is a quantum circuit, directly return that.
    # Otherwise, go through the list
    if not isinstance(circ, QuantumCircuit):
        circ = circ.lower()

        if circ == 'zzfeaturemap':
            circ = ZZFeatureMap(num_qubits, reps=1, entanglement='linear')

        elif circ == 'twolocal':
            circ = TwoLocal(num_qubits, 'ry', 'cx', 'linear', reps=1, skip_final_rotation_layer=True)

        elif circ == 'twolocalx':
            circ = TwoLocal(num_qubits, 'rx', 'cx', 'linear', reps=1, skip_final_rotation_layer=True, name = "TwoLocalX")

        elif circ == 'twolocalz':
            circ = TwoLocal(num_qubits, 'rz', 'cx', 'linear', reps=1, skip_final_rotation_layer=True, name="TwoLocalZ")

        elif circ == 'zzfeaturemap_ring':
            circ = ZZFeatureMap(num_qubits, reps=1, entanglement='circular')

        elif circ == 'twolocal_ring':
            circ = TwoLocal(num_qubits, 'ry', 'cx', 'circular', reps=1, skip_final_rotation_layer=True)

        elif circ == 'zzfeaturemap_full':
            circ = ZZFeatureMap(num_qubits, reps=1, entanglement='full')

        elif circ == 'twolocal_full':
            circ = TwoLocal(num_qubits, 'ry', 'cx', 'full', reps=1, skip_final_rotation_layer=True)

        elif circ == 'twolocal_plus':
            circ = TwoLocal(num_qubits, ['rz', 'ry'], 'cx', 'linear', reps=1, skip_final_rotation_layer=True, name = 'TwoLocal_plus')

        elif circ == 'twolocal_plus_ring':
            circ = TwoLocal(num_qubits, ['rz', 'ry'], 'cx', 'circular', reps=1, skip_final_rotation_layer=True, name='TwoLocal_plus')

        elif circ == 'twolocal_plus_full':
            circ = TwoLocal(num_qubits, ['rz', 'ry'], 'cx', 'full', reps=1, skip_final_rotation_layer=True, name='TwoLocal_plus')

        elif circ == 'twolocal_parametric2q':
            circ = TwoLocal(num_qubits, 'ry', 'crz', 'linear', reps=1, skip_final_rotation_layer=True, name='TwoLocal_parametricRz')

        elif circ == 'twolocal_parametric2q_ring':
            circ = TwoLocal(num_qubits, 'ry', 'crz', 'circular', reps=1, skip_final_rotation_layer=True, name='TwoLocal_parametricRz')

        elif circ == 'twolocal_parametric2q_full':
            circ = TwoLocal(num_qubits, 'ry', 'crz', 'full', reps=1, skip_final_rotation_layer=True, name='TwoLocal_parametricRz')

        elif circ == 'twolocal_h_parametric2q':
            circ = TwoLocal(num_qubits, 'h', 'crx', 'linear', reps=1, skip_final_rotation_layer=True, name='TwoLocal_h_parametricRz')

        elif circ == 'twolocal_h_parametric2q_ring':
            circ = TwoLocal(num_qubits, 'h', 'crx', 'circular', reps=1, skip_final_rotation_layer=True, name='TwoLocal_h_parametricRz')

        elif circ == 'twolocal_h_parametric2q_full':
            circ = TwoLocal(num_qubits, 'h', 'crx', 'full', reps=1, skip_final_rotation_layer=True, name='TwoLocal_h_parametricRz')

        elif circ == 'circuit15':
            circ = circuit15(num_qubits, num_reps=1, barrier=False)

        elif circ == 'circuit12':
            circ = circuit12(num_qubits, num_reps=1, piramidal=True, barrier=False)

        elif circ == 'circuit9':
            circ = circuit9(num_qubits, num_reps=1, barrier = False)

        elif circ == 'circuit10':
            circ = circuit10(num_qubits, num_reps=1, barrier=False)

        elif circ == 'circuit1':
            circ = circuit1(num_qubits, num_reps = 1, barrier = False)

        elif circ == 'identity':
            circ = identity(num_qubits)

        elif circ == 'circuit_adjm':
            circ == circuit_adjm(num_qubits, mat_adj, layer_ry, num_reps=1, barrier=False)

        else:
            raise ValueError(f'Circuit {circ} is not implemented.')

    return circ


