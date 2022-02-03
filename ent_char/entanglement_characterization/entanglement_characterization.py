# Import necessary modules
import sys
from qcomps import run_simulation
from qcomps.qk_utils import qk_transpilation_params
from qcomps import TNObsTensorProduct, TNObservables, QCOperators, QCConvergenceParameters
from circuits import ring_circ, Abbas_QNN
import numpy as np
import matplotlib.pyplot as plt

import qiskit as qk
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit.quantum_info.states.utils import partial_trace
from qiskit.quantum_info.states.measures import entropy
from qiskit.quantum_info import DensityMatrix
from qiskit import Aer, transpile
sim_bknd = Aer.get_backend('statevector_simulator')

seed = 21
np.random.seed(seed)

def run_mps(qc, max_bond_dim=10):
    """
    Runs a quantum circuit (parameters already numerical) with MPS.
    """

    conv_params = QCConvergenceParameters(max_bond_dim)
    trans_params = qk_transpilation_params(linearize=True)

    results = run_simulation(qc, convergence_parameters=conv_params,
                             transpilation_parameters=trans_params, do_entanglement=True,
                             approach='PY', save_mps='N', do_statevector=True)

    return results


def run_circuit(qc, parameters, max_bond_dim=10):
    """
    Assigns parameters to the quantum circuits and runs it with python-MPS.
    """

    qc = qc.assign_parameters(parameters)
    results = run_mps(qc, max_bond_dim=max_bond_dim)
    return results


def haar_entanglement(num_qubits, num_A, log_base = 'e'):
    """
    Entanglement entropy of a random pure state (haar distributed), taken from Commun. Math. Phys. 265, 95–117 (2006), Lemma II.4.
    Considering a system of num_qubits, bi-partition it in system A with num_A qubits, and B with the rest. 
    Formula applies for num_A \leq num_B.
    """

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
    Evaluates the expected value of the entanglement bond if the states were Haar distributed.
    """

    entanglement_bond = [haar_entanglement(num_qubits, i) for i in range(1, int(num_qubits / 2)+1)]
    
    # Just for fixing shapes in plotting.
    if num_qubits % 2 == 0:
        entanglement_bond = entanglement_bond + entanglement_bond[::-1][1:]
    else:
        entanglement_bond = entanglement_bond + entanglement_bond[::-1]
        
    return entanglement_bond


def entanglement_entropy(statevector, idx_to_trace=None):
    """
    Entanglement entropy of subsystem of a pure state. 
    """
    # Construct density matrix
    rho = np.outer(statevector, np.conjugate(statevector))
    # Trace out part of the system (indicated by idx_to_trace, following Qiskit indexing)
    partial_rho = partial_trace(rho, idx_to_trace)
    ent_entropy = entropy(partial_rho, base = np.e)

    return ent_entropy

def entanglement_bond(statevector):
    """
    Evaluates the entanglement entropies for cuts in a 1-d quantum system, as in MPS, given a pure state of num_qubits qubits.
    """
    num_qubits = int(np.log2(len(statevector)))
    
    # Cut along the first half of the bond
    res1 = [entanglement_entropy(statevector, idx_to_trace=range(i+1, num_qubits)) for i in range(int(num_qubits/2))]
    
    # Cut along the second half
    res2 = [entanglement_entropy(statevector, idx_to_trace=range(0, i)) for i in range(1+int(num_qubits/2), num_qubits)]
    res = res1 + res2

    return np.array(res)

def mps_simulation(qc, random_params):
    """"
    Simulation using MPS to study bond entanglement.
    """
        
    mps_results_list = []
    for idx, params in enumerate(random_params):
        print(f"Run {idx}/{len(random_params)}", end="\r")
        qc = transpile(qc, sim_bknd)
        mps_results = run_circuit(qc, params, max_bond_dim=25)
        mps_results_list.append(mps_results)

    mps_entanglement = np.array([res.entanglement for res in mps_results_list])
    ent_means = np.mean(mps_entanglement, axis=0)
    ent_std = np.std(mps_entanglement, axis=0)

    return ent_means, ent_std

def aer_simulation(qc, random_params):
    """
    Simulation using Qiskti Aer to study bond entanglement.
    """

    qk_results_list = []
    for idx, params in enumerate(random_params):
        print(f"Run {idx}/{len(random_params)}", end="\r")
        qc_t = qc.decompose()
        qk_results = sim_bknd.run(qc_t.assign_parameters(params))
        qk_results_list.append(np.asarray(qk_results.result().get_statevector()))
    
    aer_ent = np.array([entanglement_bond(state) for state in qk_results_list])
    ent_means = np.mean(aer_ent, axis=0)
    ent_std = np.std(aer_ent, axis=0)

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

def main(num_qubits = 4, num_reps = 2, ansatz = 'QNN', alternate = False, backend = 'MPS'):

    data = {'num_qubits': num_qubits,
            'num_reps': num_reps,
            'backend': backend,
            'alternate': alternate,
            'ansatz': ansatz}
    logger(data)

    ######################################################
    # CHOOSE CIRCUIT ANSATZ
    qc = qk.QuantumCircuit(num_qubits)
    if ansatz == 'ring':
        qc = qc.compose(ring_circ(num_qubits, num_reps = num_reps, barrier = False)) # Ring circ
    if ansatz == 'qnn':
        qc = qc.compose(Abbas_QNN(num_qubits, reps=num_reps, alternate=alternate, barrier=True)) # Abbas QNN
    else:
        raise TypeError(f"{ansatz} not implemented")
    

    ######################################################
    # GENERATE RANDOM PARAMETERS (both inputs and weights)
    trials = 1_00
    random_params = np.pi * np.random.rand(trials, len(qc.parameters))

    ######################################################
    # SIMULATION WITH MPS or Aer
    if backend == 'MPS':    
        ent_means, ent_std = mps_simulation(qc, random_params)
    elif backend == 'Aer':   
        ent_means, ent_std = aer_simulation(qc, random_params)
    else:
        raise TypeError(f"Backend {backend} not available")

    ######################################################
    # ENTANGLEMENT STUDY
    print("Measured entanglement = ", ent_means)

    # Expected entanglement accross cut lines if Haar distributed
    ent_haar = haar_bond_entanglement(num_qubits)
    print("Haar entanglement at bond = ", ent_haar)

    ######################################################
    # PLOT
    if False:
        fig = plt.figure(figsize=(8, 5))

        plt.title(f"{ansatz}, alternated = {alternate}, reps = {num_reps}, n_params = {len(random_params[0])}")
        plt.xticks(range(num_qubits))
        plt.ylabel("Entanglement Entropy")
        plt.xlabel("Bond entanglement")
        plt.plot(range(1, num_qubits), ent_haar, ls='--', color='red', marker='x')
        plt.errorbar(range(1, num_qubits), ent_means, yerr=ent_std)

        #qc.decompose().draw()

        plt.show()

    return ent_means, ent_std, ent_haar


if __name__ == '__main__': 

    # Quantum Cirucit structure
    num_qubits = 10
    num_reps = 6
    alternate = True
    ansatz = 'qnn'

    # Choose simulation backend
    #backend = 'MPS'
    backend = 'Aer'

    main(num_qubits, num_reps, ansatz=ansatz, backend=backend, alternate=alternate)
