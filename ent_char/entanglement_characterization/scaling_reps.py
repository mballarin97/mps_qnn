# Import necessary modules
import os
import sys

from circuits import ring_circ, Abbas_QNN, general_qnn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress

import qiskit as qk
from qiskit.circuit.library import ZZFeatureMap, TwoLocal

from entanglement_characterization import main as ent_bonds

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def is_close(measured, theory):
    dist = np.linalg.norm(theory-measured)
    dist1 = np.mean(np.abs(ent_haar-ent_meas)) 
    # Sum of entanglement
    dist2 = np.sum(theory) - np.sum(measured)
    return dist2, dist1

def create_ansatz(num_qubits, num_reps, alternate):
    # Select Circuit
    #feature_map = ring_circ(num_qubits, num_reps=num_reps, barrier=False)[0]  # Ring circ
    #ansatz = Abbas_QNN(num_qubits, reps = num_reps, alternate = alternate, barrier = True) # AbbassQNN
    # SPECIFY YOUR OWN VQC, with a feature map and variational ansatz.
    # Be sure that they use just one rep, as they are used as building blocks to build the full circuit.
    # Inputs: Parameters in the feature_map are considered like inputs, so are equal in each layer.
    # Weights: Parameters in the var_ansata are considered trainable variables, so are different in each layer.
    feature_map = ZZFeatureMap(num_qubits, reps=1, entanglement = entanglement_map)
    var_ansatz = TwoLocal(num_qubits, 'ry', 'cx', entanglement_map, reps=1, insert_barriers=True, skip_final_rotation_layer=True)

    ansatz = general_qnn(num_reps, feature_map=feature_map, var_ansatz=var_ansatz, alternate=alternate, barrier=False)

    #print(ansatz[0].decompose())

    return ansatz

seed = 34
np.random.seed(seed)

alternate = True
backend = 'Aer'
entanglement_map = "linear"

num_qubits = list(range(2, 11))
eps = 0.05

r_star = []
for idx, nq in enumerate(num_qubits):
    for num_reps in range(int(nq/2)+1, 2*nq+5):
        print(f"Run {idx} of {len(num_qubits)}, num_qubits = {nq} of {max(num_qubits)}, reps = {num_reps}", end="\r")
        
        ansatz = create_ansatz(nq, num_reps, alternate)
        
        with HiddenPrints():
            ent_meas, _, ent_haar = ent_bonds(nq, num_reps, ansatz=ansatz, backend=backend, alternate=alternate)
        dist = is_close(ent_meas, ent_haar)
        
        if dist[0] < eps:
            print(f"\nDone at {num_reps}! Values:", np.round(ent_meas,5), np.round(ent_haar,5), np.round(dist,5), "\n")
            r_star.append(num_reps)
            break

print("\n")
print(r_star)

# Fit
x = np.array(num_qubits)
y = np.array(r_star)
res = linregress(x, y)

path = f"./data/optimal_reps/"
np.savez(path+f"{max(num_qubits)}_{entanglement_map}", x=x, y=y, q=res.intercept, m=res.slope)

# Plot
import matplotlib
matplotlib.rcParams['text.usetex'] = True

fig = plt.figure(figsize=(8,5))

plt.title(fr"AbbassQNN, $\varepsilon={eps}$")
plt.xlabel("Number of qubits")
plt.ylabel(fr"Optimal repetition, $r^*$")

plt.xticks(range(min(num_qubits), max(num_qubits)+1))
plt.yticks(range(min(r_star), max(r_star)+1))

plt.scatter(x, y, marker = '.')
plt.plot(x, res.intercept + res.slope * x, 'r', ls='--',
         label=f'Fit slope = {res.slope}')

plt.legend()

plt.tight_layout()
plt.show()


            


