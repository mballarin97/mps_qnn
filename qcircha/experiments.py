# This code is part of qcircha.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os

from qcircha.circuit_selector import pick_circuit
from qcircha.entanglement_characterization import entanglement_characterization
from qcircha.utils import removekey

__all__ = ['entanglement_scaling', 'ent_vs_reps', 'compute_bond_entanglement']

def entanglement_scaling(max_num_qubits = 10, feature_map='ZZFeatureMap', var_ansatz='TwoLocal',
    backend = 'Aer', path = './data/ent_scaling/', alternate = True, max_bond_dim=1024):
    """
    Study of the total entanglement in the MPS state, varying the number of qubits,
    and save them in `path` in a file with the format "%Y-%m-%d_%H-%M-%S".

    Parameters
    ----------
    max_num_qubits : array-like or int, optional
        If int, maximum number of qubits. If array-like, interested range of qubits. by default 10
    feature_map : str or :py:class:`QuantumCircuit`, optional
        Type of feature map. Available options in the description. If a
        :py:class:`QuantumCircuit` it is used instead of the default ones.
        Default to 'ZZFeatureMap'.
    var_ansatz : str or :py:class:`QuantumCircuit`, optional
        Type of feature map. Available options in the description. If a
        :py:class:`QuantumCircuit` it is used instead of the default ones.
        Default to 'TwoLocal'.
    backend : str, optional
        Computational backend. Possible: 'Aer', 'MPS'. by default 'Aer'
    path : str, optional
        path for saving data, by default './data/ent_scaling/'
    alternate : bool, optional
        If the feature map and the ansatz should be reproduced in an alternate
        arrangement, by default True
    max_bond_dim : int, optional
        Maximum bond dimension for MPS backend. Ignored if the backend is Aer, by default 2

    Returns
    -------
    None
    """
    if isinstance(max_num_qubits, int):
        qubits_range = np.arange(4, max_num_qubits+1, 2, dtype=int)
    else:
        qubits_range = max_num_qubits

    ent_data = []
    for nqubits in qubits_range:
        tmp = ent_vs_reps(nqubits, feature_map=feature_map, var_ansatz=var_ansatz,
            backend=backend, alternate=alternate, max_bond_dim=max_bond_dim)
        ent_data.append(tmp)

    # Save data
    # path = './data/ent_scaling/'
    if not os.path.isdir(path):
        os.mkdir(path) 

    timestamp = time.localtime()
    save_as = time.strftime("%Y-%m-%d_%H-%M-%S", timestamp) + '_' + str(np.random.randint(0, 1000))
    name = os.path.join(path, save_as)

    # Save details of the ansatz
    ansatz = pick_circuit(2, 2, feature_map = feature_map, var_ansatz = var_ansatz, alternate=alternate)
    meta = dict({"max_num_qubits": max_num_qubits, "backend": backend})
    circ_data = removekey(ansatz.metadata, ["num_qubits", "num_reps", "params"])
    meta.update(circ_data)  # add metadata from the ansatz

    # Eliminate dash in circuit names, if any
    fmap = meta['fmap'].split('-')[0]
    var_ansatz = meta['var_ansatz'].split('-')[0]
    meta['fmap'] = fmap
    meta['var_ansatz'] = var_ansatz
    
    with open(name+'.json', 'w') as file:
        json.dump(meta, file, indent=4)
    
    ent_data = np.array(ent_data, dtype=object)
    np.save(name, ent_data, allow_pickle=True)


def ent_vs_reps(num_qubits, feature_map='ZZFeatureMap', var_ansatz='TwoLocal', backend = 'Aer', alternate = True, max_bond_dim=1024):
    """
    Evaluate the total entanglement (sum of entanglement accross bipartitions) in the MPS quantum state, 
    for various repetitions of the ansatz, for a fixed number of qubits.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit
    feature_map : str or :py:class:`QuantumCircuit`, optional
        Type of feature map. Available options in the description. If a
        :py:class:`QuantumCircuit` it is used instead of the default ones.
        Default to 'ZZFeatureMap'.
    var_ansatz : str or :py:class:`QuantumCircuit`, optional
        Type of feature map. Available options in the description. If a
        :py:class:`QuantumCircuit` it is used instead of the default ones.
        Default to 'TwoLocal'.
    backend : str, optional
        Computational backend. Possible: 'Aer', 'MPS'. by default 'Aer'
    alternate : bool, optional
        If the circuit structure for feature_map/ansatz is alternated (True) or
        we apply first all the feature maps and then all the ansatz (False). Default to True
    max_bond_dim : int, optional
        Maximum bond dimension for MPS backend. Ignored if the backend is Aer, by default 2

    Returns
    -------
    float
        Sum of the average entanglement along the bipartitions
    float
        Sum of the standard deviation entanglement along the bipartitions
    float
        Sum of the Haar entanglement along the bipartitions
    """
    
    ent_list, _ = compute_bond_entanglement(num_qubits, feature_map=feature_map, var_ansatz=var_ansatz, 
                                            backend = backend, alternate = alternate, 
                                            max_bond_dim = max_bond_dim)

    # Total Entanglement, sum accorss all bonds for a fixed repetition
    tot_ent_per_rep = np.sum(ent_list[: , 0, :], axis = 1) # 

    # Std deviation propagation for Total Entanglement, sum accorss all bonds for a fixed repetition
    tot_ent_per_rep_std = np.sqrt(np.sum(np.array(ent_list[:, 1, :])**2, axis=1))

    # Total Haar Entanglement, sum accorss all bonds for a fixed repetition
    haar_ent = np.sum(ent_list[:, 2, :], axis=1) 
   
    return tot_ent_per_rep, tot_ent_per_rep_std, haar_ent[0]

def compute_bond_entanglement(num_qubits, feature_map='ZZFeatureMap', var_ansatz='TwoLocal', alternate = True,
    backend = 'Aer', plot = False, max_bond_dim = None):
    """
    Evaluate entanglement entropy accross bonds, varying the repetitions of the variational ansatz,
    for a fixed number of qubits.

    Parameters
    ----------
    num_qubits : int
        Number of qubits
    feature_map : str or :py:class:`QuantumCircuit`, optional
        Type of feature map. Available options in the description. If a
        :py:class:`QuantumCircuit` it is used instead of the default ones.
        Default to 'ZZFeatureMap'.
    var_ansatz : str or :py:class:`QuantumCircuit`, optional
        Type of feature map. Available options in the description. If a
        :py:class:`QuantumCircuit` it is used instead of the default ones.
        Default to 'TwoLocal'.
    alternate : bool, optional
        If the feature map and the variational ansatz should be alternated in the
        disposition (True) or if first apply ALL the feature map repetitions and
        then all the ansatz repetitions (False). Default to True.
    backend : str, optional
        Computational backend. Possible 'MPS' or 'Aer', by default 'Aer'
    plot : bool, optional
        If True, perform some plots at the end, by default False
    max_bond_dim : int, optional
        maximum bond dimension if we use the MPS backend, by default None

    Returns
    -------
    TODO array-like of floats
        list of the results of ent_char. We need to improve this
    array-like of floats
        Maximum entanglement given by a completely mixed state
    """

    ####################################################################
    # Entanglement in circuit and haar
    max_rep = int(1.5*num_qubits)

    ent_list = []
    for num_reps in range(1, max_rep):
        print(f"\n__Reps {num_reps}/{max_rep}")

        # Pick a PQC (modify the function)
        ansatz = pick_circuit(num_qubits, num_reps, feature_map=feature_map,
                              var_ansatz=var_ansatz, alternate = alternate)
        
        # If MPS, add max_bond_dim to circuit's metadata
        if backend == "MPS":
            data = ansatz.metadata 
            data['max_bond_dim'] = max_bond_dim
            ansatz.metadata = data

        # Run simulation and save result
        tmp = entanglement_characterization(ansatz=ansatz, backend=backend)
        ent_list.append(tmp[:-1])

    ent_list = np.array(ent_list)
    
    ####################################################################
    # MAX ENTANGLEMENT for a system of dimension d, it is d (completely mixed state).
    max_ent = [-np.log(1/2**(min(n, num_qubits-n))) for n in range(1, num_qubits)]
    
    ####################################################################
    # Plot
    if plot:
        print(ent_list)
        fig = plt.figure(figsize=(9.6, 6))
        plt.title(f"{var_ansatz[1]}, alternated = {alternate}")
        plt.xticks(range(num_qubits))
        plt.ylabel("Entanglement Entropy")
        plt.xlabel("Bond index cut")

        for idx, data in enumerate(ent_list):
            plt.errorbar(range(1, num_qubits), data[0], yerr=0*data[1], label=f"Rep {idx+1}", marker = 'o')
        
        plt.plot(range(1, num_qubits), ent_list[0, 2], ls='-.', color='red', marker='v', label="Haar")
        plt.plot(range(1, num_qubits), max_ent, ls='--', marker='.', label="Maximum entanglement")
        plt.legend()
        plt.tight_layout()

        plt.show()

    return ent_list, max_ent
    