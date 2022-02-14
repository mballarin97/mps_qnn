from qiskit import Aer
import qiskit as qk
from qiskit.visualization import plot_histogram

from circuits import *

from run_simulations import main as run_qnn

from tqdm import tqdm
import time
import json
import os

import numpy as np
import scipy as sp
import scipy.linalg

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

def removekey(d, keys):
    r = dict(d)
    for key in keys:
        del r[key]
    return r

def haar_discrete(xmin, delta, n):
    return (1. - xmin)**(n-1) - (1 - xmin - delta)**(n-1)

def kl_divergence(p, q):
    #return np.sum(p * np.log(p/q))
    return sp.stats.entropy(p, q, base=np.e)

def eval_kl(fidelities, n_bins=20, num_qubits=4):

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


def main(repetitions, fmap=None, var_ansatz=None, backend= 'Aer', path='./data/expr/', plot = False, save = False):

    if isinstance(repetitions, int):
        reps = range(1, repetitions)
    else:
        reps = repetitions
    
    # Generate random states from QNN 
    st_list = []
    for num_reps in reps:
        ansatz = general_qnn(num_reps, feature_map=feature_map, var_ansatz=var_ansatz, alternate=alternate, barrier=False)
        _, _, _, statevectors = run_qnn(ansatz, backend=backend, get_statevector=True)
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
    n_bins = 20
    num_qubits = ansatz.metadata['num_qubits']
    expressibility = [eval_kl(data, n_bins=n_bins, num_qubits = num_qubits)[0] for data in res]

    if save == True:
        # Save data
        if not os.path.isdir(path):
            os.mkdir(path)

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

        plt.ylabel(r"$Expr. D_{KL}$")
        plt.xlabel("Repetitions")

        plt.yscale('log')
        plt.plot(reps, expressibility, marker='o', ls='--')
        plt.tight_layout()
        plt.show()

    return expressibility


if __name__ == '__main__':
    
    # Fixing seed for reproducibility
    seed = 120
    np.random.seed(seed)

    num_qubits = 6
    feature_map = ZZFeatureMap(num_qubits, reps=1, entanglement='linear')  
    var_ansatz = TwoLocal(num_qubits, 'ry', 'cx', 'linear', reps=1, insert_barriers=True, skip_final_rotation_layer=True)
    # ZZFeatureMap(num_qubits, reps=1, entanglement='linear')
    # TwoLocal(num_qubits, 'ry', 'cx', 'linear', reps=1, insert_barriers=True, skip_final_rotation_layer=True)
    # qk.QuantumCircuit(num_qubits, name="Id", metadata={'entanglement_map': None}) # Identiy

    alternate = True
    backend = 'Aer'
    repetitions = 15
    path = './data/expr/'
    main(repetitions, fmap=feature_map, var_ansatz=var_ansatz, backend=backend, 
         path=path, plot = True, save = True)
