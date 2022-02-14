# Import necessary modules
import sys
import time

from qcomps import run_simulation
from qcomps.qk_utils import qk_transpilation_params
from qcomps import TNObsTensorProduct, TNObservables, QCOperators, QCConvergenceParameters
from circuits import ring_circ, Abbas_QNN, general_qnn
import numpy as np
import matplotlib.pyplot as plt

import qiskit as qk
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit.quantum_info.states.utils import partial_trace
from qiskit.quantum_info.states.measures import entropy
from qiskit.quantum_info import DensityMatrix
from qiskit import Aer, transpile
sim_bknd = Aer.get_backend('statevector_simulator')

def run_mps(qc, max_bond_dim=1024):
    """
    Runs a quantum circuit (parameters already numerical) with MPS.
    """

    conv_params = QCConvergenceParameters(max_bond_dim)
    trans_params = qk_transpilation_params(linearize=True)

    results = run_simulation(qc, convergence_parameters=conv_params,
                             transpilation_parameters=trans_params, do_entanglement=True,
                             approach='PY', save_mps='N', do_statevector=False)

    return results

def run_circuit(qc, parameters, max_bond_dim=1024):
    """
    Assigns parameters to the quantum circuits and runs it with python-MPS.
    """

    qc = qc.assign_parameters(parameters)
    results = run_mps(qc, max_bond_dim=max_bond_dim)
    return results

def harmonic(n):
    """
    Approximation of the Harmonic series H(n) = \sum_{k=1}^n 1/k ~ ln(n) + gamma + O(1/n)
    """
    return np.log(n) + np.euler_gamma

def approx_haar_entanglement(num_qubits, num_A):
    """
    Approximate expression of the haar_entanglement function, very accurate for num_qubits > 10 (error < 1%).
    """
    num_A = min([num_A, num_qubits - num_A])
    d = (2**num_A - 1) / (2**(num_qubits - num_A + 1))
    ent = harmonic(2**num_qubits) - harmonic(2**(num_qubits - num_A)) - d
    return ent

def haar_entanglement(num_qubits, num_A, log_base = 'e'):
    """
    Entanglement entropy of a random pure state (haar distributed), taken from Commun. Math. Phys. 265, 95–117 (2006), Lemma II.4.
    Considering a system of num_qubits, bi-partition it in system A with num_A qubits, and B with the rest. 
    Formula applies for num_A \leq num_B.
    """

    # Pick the smallest bi-partition
    num_A = min([num_A, num_qubits - num_A])

    dim_qubit = 2  # qubit has dimension 2
    da = dim_qubit ** num_A  # dimension of partition A
    db = dim_qubit ** (num_qubits - num_A)  # dimension of partition B
    
    ent = np.sum([1.0 / j for j in range(1+db, da * db + 1)])
    ent -= (da - 1) / (2*db)

    if log_base == '2': 
        ent *= 1 / np.log(2)

    return ent

def haar_bond_entanglement(num_qubits):
    """
    Evaluates the expected value of the entanglement at each bond if the states were Haar distributed.
    """

    if num_qubits < 20:
        entanglement_bonds = [haar_entanglement(num_qubits, i) for i in range(1, num_qubits)]
    else:
        entanglement_bonds = [approx_haar_entanglement(num_qubits, i) for i in range(1, num_qubits)]
    
    return entanglement_bonds

def entanglement_entropy(statevector, idx_to_trace=None):
    """
    Entanglement entropy of subsystem of a pure state. Given a statevector (i.e. pure state), builds the density matrix, 
    and traces out some systems. Then eveluates Von Neumann entropy using Qiskit's implementation. 
    Be consistent with the base of the logarithm. 
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
    """
    num_qubits = int(np.log2(len(statevector)))
    
    # Cut along the first half of the bond
    res1 = [entanglement_entropy(statevector, idx_to_trace=range(i+1, num_qubits)) for i in range(int(num_qubits/2))]
    
    # Cut along the second half
    res2 = [entanglement_entropy(statevector, idx_to_trace=range(0, i)) for i in range(1+int(num_qubits/2), num_qubits)]
    res = res1 + res2

    return np.array(res)

def mps_simulation(qc, random_params, max_bond_dim=1024):
    """"
    Simulation using MPS to study bond entanglement.
    """
        
    mps_results_list = []
    for idx, params in enumerate(random_params):
        print(f"Run {idx}/{len(random_params)}", end="\r")
        qc = transpile(qc, sim_bknd)
        mps_results = run_circuit(qc, params, max_bond_dim=max_bond_dim)
        mps_results_list.append(mps_results)

    mps_entanglement = np.array([res.entanglement for res in mps_results_list])
    ent_means = np.mean(mps_entanglement, axis=0)
    ent_std = np.std(mps_entanglement, axis=0)

    return ent_means, ent_std

def aer_simulation(qc, random_params, get_statevector = False):
    """
    Simulation using Qiskit Aer to study bond entanglement.
    """

    qk_results_list = []
    for idx, params in enumerate(random_params):
        print(f"Run {idx}/{len(random_params)}", end="\r")
        qc_t = qc.decompose()
        qk_results = sim_bknd.run(qc_t.assign_parameters(params))
        qk_results_list.append(np.asarray(qk_results.result().get_statevector()))
    
    print("\nPartial tracing...")
    now = time.time()
    aer_ent = np.array([entanglement_bond(state) for state in qk_results_list])
    print(f" >> Ended in {time.time()-now}")
    ent_means = np.mean(aer_ent, axis=0)
    ent_std = np.std(aer_ent, axis=0)

    if get_statevector == True:
        return ent_means, ent_std, qk_results_list
    else:
        return ent_means, ent_std

def logger(data):
    """
    Printing data.
    """

    print("===============================")
    print("SIMULATION DETAILS")
    for k,v in zip(data.keys(), data.values()):
        print(f"{k} = {v}", end="\n")
    print("===============================")

def main(ansatz = None, backend = 'Aer', get_statevector = False):

    metadata = ansatz.metadata
    logger(metadata)

    num_qubits = metadata['num_qubits']
    num_reps = metadata['num_reps']
    alternate = metadata['alternate']

    try:
        max_bond_dim = metadata['max_bond_dim']
    except: 
        max_bond_dim = 1_024
             

    ######################################################
    # GENERATE RANDOM PARAMETERS (both inputs and weights)
    trials = 1_00
    random_params = np.pi * np.random.rand(trials, len(ansatz.parameters))

    ######################################################
    # SIMULATION WITH MPS or Aer
    if backend == 'MPS':    
        ent_means, ent_std = mps_simulation(ansatz, random_params, max_bond_dim)
    elif backend == 'Aer':  
        if get_statevector == True:
            ent_means, ent_std, statevectors = aer_simulation(ansatz, random_params, get_statevector=get_statevector)
        else: 
            ent_means, ent_std = aer_simulation(ansatz, random_params)
    else:
        raise TypeError(f"Backend {backend} not available")

    ######################################################
    # ENTANGLEMENT STUDY
    print("Measured entanglement =     ", np.round(ent_means, 4))

    # Expected entanglement accross cut lines if Haar distributed
    ent_haar = haar_bond_entanglement(num_qubits)
    print("Haar entanglement at bond = ", np.round(ent_haar, 4))

    ######################################################
    # PLOT
    if False:
        fig = plt.figure(figsize=(8, 5))

        plt.title(f"{ansatz}, alternated = {alternate}, reps = {num_reps}, n_params = {len(ansatz.parameters)}")
        plt.xticks(range(num_qubits))
        plt.ylabel("Entanglement Entropy")
        plt.xlabel("Bond entanglement")
        plt.plot(range(1, num_qubits), ent_haar, ls='--', color='red', marker='x')
        plt.errorbar(range(1, num_qubits), ent_means, yerr=ent_std)

        #qc.decompose().draw()

        plt.show()

    if get_statevector== True:
        return ent_means, ent_std, ent_haar, statevectors
    else:
        return ent_means, ent_std, ent_haar


if __name__ == '__main__': 

    # Fixing seed for reproducibility
    seed = 34
    np.random.seed(seed)

    # Quantum Cirucit structure
    num_qubits = 4
    num_reps = 4
    alternate = True

    # Select a feature map and a variational block
    # Reps: Be sure that they use just one rep, as they are used as building blocks to build the full circuit. 
    # Inputs: Parameters in the feature_map are considered like inputs, so are equal in each layer.
    # Weights: Parameters in the var_ansata are considered trainable variables, so are different in each layer.
    #ansatz = ring_circ(num_qubits, num_reps=num_reps, barrier=False)  # Ring circ
    #ansatz = Abbas_QNN(num_qubits, reps = num_reps, alternate = alternate, barrier = True) # AbbassQNN
    feature_map = ZZFeatureMap(num_qubits, reps=1, entanglement='linear')
    var_ansatz = TwoLocal(num_qubits, 'ry', 'cx', 'linear', reps=1, insert_barriers=True, skip_final_rotation_layer=True)
    ansatz = general_qnn(num_reps, feature_map = feature_map, var_ansatz = var_ansatz, alternate = alternate, barrier = False)

    # Choose simulation backend
    #backend = 'MPS'
    backend = 'Aer'

    main(ansatz, backend=backend)
