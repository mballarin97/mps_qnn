import numpy as np
from qcircha import entanglement_scaling, compute_bond_entanglement
from qiskit.circuit.library import TwoLocal

def main():

    # SELECT STRUCTURE
    alternate = True

    # SIMULATION BACKEND
    backend = 'Aer'
    # backend = 'MPS'

    # TOTAL ENTANGLEMENT SCALING    
    max_num_qubits = 12 # or np.arange(30, 51, 10)
    entanglement_scaling(max_num_qubits, 
                         feature_map='TwoLocal', var_ansatz="TwoLocal",
                         alternate = alternate, backend = backend,
                         max_bond_dim = 1024, path='./data/ent_scaling/')


    # ENTANGLEMENT ACROSS BIPARTITIONS
    # num_qubits = 5
    # compute_bond_entanglement(num_qubits, 
    #                          feature_map='ZZFeatureMap', var_ansatz='TwoLocal', 
    #                          alternate=True, backend = backend, 
    #                          plot=True, max_bond_dim=None)

if __name__ == '__main__':

    #seed = 34
    #np.random.seed(seed)
    main()