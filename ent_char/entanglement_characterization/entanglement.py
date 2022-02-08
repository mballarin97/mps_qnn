import numpy as np
import matplotlib.pyplot as plt
import time
import json

from qiskit.circuit.library import ZZFeatureMap, TwoLocal

from run_simulations import main as ent_char
from circuits import *

def removekey(d, keys):
    r = dict(d)
    for key in keys:
        del r[key]
    return r

def entanglement_scaling(max_num_qubits = 10, backend = 'Aer', alternate = True):
    """
    Study of the total entanglement in the MPS state, varying the number of qubits. 
    """

    ent_data = []
    for nqubits in range(4, max_num_qubits+1, 2):
        tmp = ent_vs_reps(nqubits, backend=backend, alternate=alternate)
        ent_data.append(tmp)

    # Save data
    path = './data/ent_scaling/'

    timestamp = time.localtime()
    save_as = time.strftime("%Y-%m-%d_%H-%M-%S", timestamp) + '_' + str(np.random.randint(0, 1000))
    name = path + save_as

    # Save details of the ansatz
    ansatz = pick_circuit(2, 2, alternate=alternate)
    meta = dict({"max_num_qubits": max_num_qubits, "backend": backend})
    circ_data = removekey(ansatz.metadata, ["num_qubits", "num_reps", "params"])
    meta.update(circ_data)  # add metadata from the ansatz
    
    with open(name+'.json', 'w') as file:
        json.dump(meta, file, indent=4)
    
    ent_data = np.array(ent_data, dtype=object)
    np.save(name, ent_data, allow_pickle=True)

    #How to plot
    #data = np.array(data)
    #cmap = plt.get_cmap('tab10')

    #fig = plt.figure(figsize=(12, 8))
    #plt.title(f"Circuit")
    #plt.ylabel("Entanglement Entropy")
    #plt.xlabel("Bond index cut")

    #for idx, data in enumerate(ent_data):
    #    plt.plot(data[0], ls="-", marker=".", c=cmap(idx), label=f"Num_qubits = {idx+1}")
    #    plt.hlines(data[1], 0, len(ent_data[-1][0]), ls="--", color = cmap(idx), label="Haar")
    #
    #plt.legend()
    #plt.show()

def ent_vs_reps(num_qubits, backend = 'Aer', alternate = True):
    """
    Evaluate the total entanglement (sum of entanglement accross bipartitions) in the MPS quantum state, 
    for various repetitions of the ansatz, for a fixed number of qubits.
    """
    
    ent_list, _ = main(num_qubits, backend = backend, alternate = alternate)

    # Total Entanglement, sum accorss all bonds for a fixed repetition
    tot_ent_per_rep = np.sum(ent_list[: , 0, :], axis = 1) # 

    # Std deviation propagation for Total Entanglement, sum accorss all bonds for a fixed repetition
    tot_ent_per_rep_std = np.sqrt(np.sum(np.array(ent_list[:, 1, :])**2, axis=1))

    # Total Haar Entanglement, sum accorss all bonds for a fixed repetition
    haar_ent = np.sum(ent_list[:, 2, :], axis=1) 
   
    #num_reps = len(ent_list)
    #fig = plt.figure(figsize=(8,5))
    #plt.plot(range(1, num_reps+1), tot_ent_per_rep)
    #plt.hlines(haar_ent[0], 1, num_reps, ls='--', color='r')
    #plt.show()

    return tot_ent_per_rep, tot_ent_per_rep_std, haar_ent[0]


def alt_comparison(num_qubits, ansatz = None, backend = 'Aer'):
    """
    Plot joined figure of alternated vs. non alternated architecture.
    """

    res = []
    for alternate in [True, False]:
        ent_list, max_ent = main(num_qubits, alternate=alternate, backend='Aer', plot = False)
        res.append(ent_list)
    res = np.array(res)

    cmap = plt.get_cmap('tab10')

    fig = plt.figure(figsize=(12, 8))
    plt.title("Circuit")
    plt.xticks(range(num_qubits))
    plt.ylabel("Entanglement Entropy")
    plt.xlabel("Bond index cut")

    for idx, data in enumerate(res[0]):
        plt.errorbar(range(1, num_qubits), data[0], yerr=data[1], 
                    ls="-", marker=".", elinewidth=0.0, c = cmap(idx),
                    label=f"Alt, rep {idx+1}")

    for idx, data in enumerate(res[1]):
        plt.errorbar(range(1, num_qubits), data[0], yerr=data[1], 
                    ls="--", marker=".", elinewidth = 0.0, c = cmap(idx),
                    label=f"No Alt, rep {idx+1}")
    
    plt.plot(range(1, num_qubits),
             res[0, 0, 2], ls=':', color='red', marker='X', label="Haar")
    plt.plot(range(1, num_qubits), max_ent,
             ls=':', marker='D', label="Maximum entanglement")
    plt.legend()
    plt.tight_layout()
    plt.show()

def pick_circuit(num_qubits, num_reps, alternate = True):
    """
    Select a circuit with a feature map and a variational block. Examples below.
    Each block must have reps = 1, and then specify the 
    """

    # Example: Abbass-QNN
    # feature_map = ZZFeatureMap(num_qubits, reps=1, entanglement='linear')
    # var_ansatz = TwoLocal(num_qubits, 'ry', 'cx', 'linear', reps=1, insert_barriers=False, skip_final_rotation_layer=True)
    # or already defined full PQC
    # Abbas_QNN(num_qubits, reps=num_reps, alternate=alternate, barrier=True)  # Full PQC

    # Example:
    # feature_map = ZZFeatureMap(num_qubits, reps=1, entanglement='linear')
    # feature_map = piramidal_circuit(num_qubits, num_reps=1, piramidal=False, barrier=False)
    # feature_map = dummy_circ(num_qubits, num_reps = 1, barrier = True)
    # feature_map = TwoLocal(num_qubits, ['rx', 'rz'], 'cx', 'linear', reps = 1, insert_barriers=True, skip_final_rotation_layer=True)
    feature_map = circuit9(num_qubits, num_reps = 1, barrier = True)
    var_ansatz = circuit9(num_qubits, num_reps=1, barrier=True)

    # Other examples:
    # ring_circ(num_qubits, num_reps=1, barrier=False)  # Ring circ (n.15 from Kim et al.)
    # piramidal_circuit(num_qubits, num_reps=1, piramidal=True, barrier=False) # Piramidal circ (n.12 from Kim et al.)

    # Build the PQC
    ansatz = general_qnn(num_reps, feature_map=feature_map,
                         var_ansatz=var_ansatz, alternate=alternate, barrier=False)


    return ansatz


def main(num_qubits, alternate = True, backend = 'Aer', plot = False):
    """
    Evaluate entanglement entropy for multiple repetitions of the variational ansatz and fixed number of qubits.
    """

    # Entanglement in circuit and haar
    ent_list = []
    max_rep = int(1.5*num_qubits)
    for num_reps in range(1, max_rep):
        print(f"\n__Reps {num_reps}/{max_rep}")

        # Pick a PQC (modify the function)
        ansatz = pick_circuit(num_qubits, num_reps, alternate = alternate)

        # Run simulation and save result
        tmp = ent_char(ansatz=ansatz, backend=backend)
        ent_list.append(tmp)

    ent_list = np.array(ent_list)
    
    ####################################################################
    # MAX ENTANGLEMENT for a system of dimension d, is d.
    max_ent = [-np.log(1/2**n) for n in range(1, int(num_qubits/2)+1)]
    # Just for fixing shapes in plotting.
    if num_qubits % 2 == 0:
        max_ent = max_ent + max_ent[::-1][1:]
    else:
        max_ent = max_ent + max_ent[::-1]
    ####################################################################

    # Plot
    if plot:
        fig = plt.figure(figsize=(8, 5))
        plt.title(f"{ansatz[1]}, alternated = {alternate}")
        plt.xticks(range(num_qubits))
        plt.ylabel("Entanglement Entropy")
        plt.xlabel("Bond index cut")
        for idx, data in enumerate(ent_list):
            plt.errorbar(range(1, num_qubits),
                         data[0], yerr=data[1], label=f"Rep {idx+1}")
        
        plt.plot(range(1, num_qubits),
                 ent_list[0, 2], ls='--', color='red', marker='x', label="Haar")
        plt.plot(range(1, num_qubits), max_ent,
                 ls='--', marker='.', label="Maximum entanglement")
        plt.legend()
        plt.tight_layout()

        #ansatz[0].decompose().draw()

        plt.show()

    return ent_list, max_ent

if __name__ == '__main__':

    seed = 43
    np.random.seed(seed)

    # Quantum Cirucit structure
    #num_qubits = 8
    alternate = True

    # Choose simulation backend
    backend = 'MPS'
    #backend = 'Aer'

    max_num_qubits = 42
    entanglement_scaling(max_num_qubits, backend = backend, alternate = alternate)

    #main(num_qubits, backend=backend, alternate=alternate)
    #ent_vs_reps(num_qubits, alternate = alternate, backend=backend)
    #alt_comparison(num_qubits, backend=backend)

    