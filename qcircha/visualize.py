"""
Visualization and plot functions
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_alternated_comparison(num_qubits, res, ansatz = None, backend = 'Aer'):
    """
    Plot joined figure of alternated vs. non alternated architecture.

    TODO:
        This is a plotting function. You don't want to call computational methods from here,
        but to load results.
    """

    """
    res = []
    for alternate in [True, False]:
        ent_list, max_ent = main(num_qubits, alternate=alternate, backend=backend, plot = False)
        res.append(ent_list)
    """
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

    ############################
    # MAXENT NOT DEFINED
    ############################
    #plt.plot(range(1, num_qubits), max_ent,
    #         ls=':', marker='D', label="Maximum entanglement")
    
    plt.legend()
    plt.tight_layout()
    plt.show()
