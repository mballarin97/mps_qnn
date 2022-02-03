from entanglement_characterization import main as ent_char
import matplotlib.pyplot as plt
import numpy as np

def alt_comparison(num_qubits, ansatz = 'qnn', backend = 'Aer'):
    """
    Plot joined figure of alternated vs. non alternated architecture. 
    """

    res = []
    for alternate in [True, False]:
        ent_list, max_ent = main(num_qubits, ansatz='qnn', alternate=alternate, backend='Aer', plot = False)
        res.append(ent_list)
    res = np.array(res)

    cmap = plt.get_cmap('tab10')

    fig = plt.figure(figsize=(12, 8))
    plt.title(f"{ansatz}")
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


def main(num_qubits, ansatz = 'qnn', alternate = True, backend = 'Aer', plot = True):
    """
    Plots entanglement entropy for multiple repetitions of a variational ansats.
    """

    # Entanglement in circuit and haar
    ent_list = []
    max_rep = num_qubits
    for num_reps in range(1, max_rep, 2):
        print(f"\n__Reps {num_reps}/{max_rep}")
        ent_list.append(ent_char(num_qubits, num_reps, ansatz=ansatz,
                        backend=backend, alternate=alternate))
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
        plt.title(f"{ansatz}, alternated = {alternate}")
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
        plt.show()

    return ent_list, max_ent



if __name__ == '__main__':

    #################################
    # Hyperparams
    num_qubits = 15
    alternate = True
    ansatz = 'qnn' # 'qnn' or 'ring'
    backend = 'Aer' # 'Aer' or 'MPS'

    #main(num_qubits, ansatz=ansatz, backend=backend, alternate=alternate)
    alt_comparison(num_qubits, ansatz=ansatz, backend=backend)

    