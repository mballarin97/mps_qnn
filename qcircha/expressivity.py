"""
Compute the expressivity of a parametrized quantum circuit by computing the KL-divergence from haar
state
"""
import time
import json
import os
import sys

import numpy as np
import scipy as sp
from qiskit import Aer

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

from qcircha.circuits import *
from qcircha.circuit_selector import pick_circuit
from qcircha.entanglement_characterization import entanglement_characterization
from qcircha.entanglement.haar_entanglement import haar_discrete
from qcircha.utils import removekey


def kl_divergence(p, q, eps = 1e-20) :
    """
    Compute the KL divergence between two probability
    distributions

    Parameters
    ----------
    p : array-like of floats
        First probability distribution
    q : array-like of floats
        Second probability distribution

    Returns
    -------
    float
        KL divergence of p and q
    """

    # Eliminate divergences (check if safe to do)
    q = np.array([max(eps, q_j) for q_j in q])
    p = np.array([max(eps, p_j) for p_j in p])

    #return np.sum(p * np.log(p/q))
    return sp.stats.entropy(p, q, base=np.e)

def eval_kl(fidelities, n_bins=20, num_qubits=4):
    """
    Evaluate KL divergence

    Parameters
    ----------
    fidelities : _type_
        _description_
    n_bins : int, optional
        _description_, by default 20
    num_qubits : int, optional
        _description_, by default 4

    Returns
    -------
    _type_
        _description_
    """

    discrete_haar = np.array([haar_discrete(x, 1/n_bins, 2 ** num_qubits) for x in np.linspace(0, 1, n_bins + 1)[:-1]])
    y, _ = np.histogram(fidelities, range=(0, 1), bins=n_bins)
    y = y / np.sum(y)

    return kl_divergence(y, discrete_haar), y, discrete_haar

def inner_products(state_list, rep):
    """
    """
    inner_p = []
    num_tests = len(state_list[0, :, 0])

    for i in range(num_tests):
        for j in range(i):
            tmp = np.abs(state_list[rep, i, :] @ np.conjugate(state_list[rep, j, :])) ** 2
            inner_p.append(tmp)
    return np.array(inner_p)


def compute_espressivity(num_qubits, repetitions, feature_map = None, var_ansatz=None, alternate = True,
                         backend='Aer', path='./data/expr/', plot=False, save=False, max_bond_dim=None):

    if isinstance(repetitions, int):
        reps = range(1, repetitions + 1)
    else:
        reps = repetitions
    
    # Generate random states from QNN 
    st_list = []
    for num_reps in reps:
        ansatz = pick_circuit(num_qubits, num_reps, feature_map=feature_map,
                              var_ansatz=var_ansatz, alternate=alternate)
        _, _, _, statevectors = entanglement_characterization(ansatz, backend=backend, get_statevector=True, 
            max_bond_dim = max_bond_dim)
        statevectors = np.array(statevectors)
        st_list.append(statevectors)
        print("")
    st_list = np.array(st_list)

    # Evaluate inner products 
    res = []    
    for idx, rep in enumerate(reps):
        res.append(inner_products(st_list, idx))
    res = np.array(res)

    # Build histogram of distribution and evaluate KL divergence with Haar 
    n_bins = 100
    num_qubits = ansatz.metadata['num_qubits']
    expressibility = [eval_kl(data, n_bins=n_bins, num_qubits = num_qubits)[0] for data in res]

    if save == True:
        # Save data
        if not os.path.isdir(path):
            os.makedirs(path)

        timestamp = time.localtime()
        save_as = time.strftime("%Y-%m-%d_%H-%M-%S", timestamp) + '_' + str(np.random.randint(0, 1000))
        name = os.path.join(path, save_as)

        # Save details of the ansatz
        meta = dict({"n_bins": n_bins, "backend": backend})
        circ_data = removekey(ansatz.metadata, ["num_reps", "params"])
        meta.update(circ_data)  # add metadata from the ansatz

        with open(name+'.json', 'w') as file:
            json.dump(meta, file, indent=4)

        expressibility = np.array(expressibility, dtype=object)
        np.save(name, expressibility, allow_pickle=True)

    if plot == True:
        fig = plt.figure(figsize=(9.6, 6))

        plt.ylim([1e-3, 1])
        plt.ylabel(r"$Expr. D_{KL}$")
        plt.xlabel("Repetitions")

        plt.yscale('log')
        plt.plot(reps, expressibility, marker='o', ls='--')
        plt.tight_layout()
        plt.show()

    return expressibility


if __name__ == '__main__':
    
    # Fixing seed for reproducibility
    # seed = 120
    # np.random.seed(seed)

    num_qubits = 6
    feature_map = 'ZZFeatureMap'
    var_ansatz = 'TwoLocal'

    alternate = True
    backend = 'Aer'
    repetitions = num_qubits
    path = './data/expr/'
    compute_espressivity(num_qubits, repetitions, feature_map = feature_map, var_ansatz=var_ansatz, backend=backend,
         path=path, plot = True, save = True)
