import numpy as np
import matplotlib.pyplot as plt
import time

from qiskit.circuit.library import ZZFeatureMap, TwoLocal

from entanglement_characterization import main as ent_char
from circuits import ring_circ, general_qnn, Abbas_QNN

def entanglement_scaling(max_num_qubits = 10, backend = 'Aer', alternate = True):

    ent_data = []
    for nq in range(4, max_num_qubits+1, 2):
        tmp = ent_vs_reps(nq, backend=backend, alternate = alternate)
        ent_data.append(tmp)

    # Save data
    path = "./data/ent_scaling/"

    timestamp = time.localtime()
    save_as = time.strftime("%Y-%m-%d_%H-%M-%S", timestamp) + '_' + str(np.random.randint(0, 1000))

    idx = np.random.randint(0, high = 5000)
    name = path+str(max_num_qubits)+save_as
    np.save(name, np.array(ent_data, dtype = object), allow_pickle=True)

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
    Evaluate the total entanglement (sum of entanglement accross bipartitions) in the quantum state, 
    for various repetitions of the ansatz, for a fixed number of qubits.
    """
    
    ent_list, _ = main(num_qubits, backend=backend, alternate=alternate)

    num_reps = len(ent_list)

    tot_ent_per_rep = np.sum(ent_list[: , 0, :], axis = 1) # sum accorss all bonds for a fixed repetition 
    tot_ent_per_rep_std = np.sum(ent_list[:, 1, :], axis=1) # sum accorss all bonds for a fixed repetition
    haar_ent = np.sum(ent_list[:, 2, :], axis=1) # sum accorss all bonds for a fixed repetition 
   
    #fig = plt.figure(figsize=(8,5))
    #plt.plot(range(1, num_reps+1), tot_ent_per_rep)
    #plt.hlines(haar_ent[0], 1, num_reps, ls='--', color='r')
    #plt.show()

    return tot_ent_per_rep, haar_ent[0]


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
    plt.title(f"Circuit")
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


def main(num_qubits, alternate = True, backend = 'Aer', plot = False):
    """
    Evaluate entanglement entropy for multiple repetitions of a variational ansats, and fixed number of qubits.
    """

    # Entanglement in circuit and haar
    ent_list = []
    max_rep = int(1.5*num_qubits)
    for num_reps in range(1, max_rep):
        print(f"\n__Reps {num_reps}/{max_rep}")

        # Select circuit
        # ansatz = Abbas_QNN(num_qubits, reps=num_reps, alternate=alternate, barrier=True)  # AbbassQNN
        feature_map = ring_circ(num_qubits, num_reps=1, barrier=False)[0]  # Ring circ

        # General QNN
        #feature_map = ZZFeatureMap(num_qubits, reps=1, entanglement='linear')
        var_ansatz = TwoLocal(num_qubits, 'ry', 'cx', 'linear', reps=1, insert_barriers=False, skip_final_rotation_layer=True)
        ansatz = general_qnn(num_reps, feature_map=feature_map, var_ansatz=var_ansatz, alternate=alternate, barrier=False)

        # Run simulation and save result
        tmp = ent_char(num_qubits, num_reps, ansatz=ansatz, backend=backend, alternate=alternate)
        ent_list.append(tmp)

    ent_list = np.array(ent_list)
    
    #################################
    # Max entanglement for a system of dimension d, is d.
    max_ent = [-np.log(1/2**n) for n in range(1, int(num_qubits/2)+1)]
    # Just for fixing shapes in plotting.
    if num_qubits % 2 == 0:
        max_ent = max_ent + max_ent[::-1][1:]
    else:
        max_ent = max_ent + max_ent[::-1]
    
    ###################################
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

    #seed = 32
    #np.random.seed(seed)

    # Quantum Cirucit structure
    #num_qubits = 8
    alternate = True

    # Choose simulation backend
    #backend = 'MPS'
    backend = 'Aer'

    max_num_qubits = 10
    entanglement_scaling(max_num_qubits, backend=backend, alternate=alternate)
    #main(num_qubits, backend=backend, alternate=alternate)
    #ent_vs_reps(num_qubits, alternate = alternate, backend=backend)
    #alt_comparison(num_qubits, backend=backend)

    