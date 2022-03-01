import numpy as np
from qcircha import entanglement_scaling, compute_bond_entanglement

def main():
    #seed = 34
    #np.random.seed(seed)

    # Quantum Cirucit structure
    num_qubits = 5
    alternate = True

    # Choose simulation backend
    #backend = 'MPS'
    backend = 'Aer'

    max_num_qubits = 12 #np.arange(30, 51, 10)
    entanglement_scaling(max_num_qubits, 
                         feature_map='TwoLocal_h_parmetric2q_full', var_ansatz="TwoLocal_full",
                         alternate = alternate, backend = backend,
                         max_bond_dim = 1024, path='./data/ent_scaling/')

    #compute_bond_entanglement(6, 
    #                          feature_map='circuit1', var_ansatz='TwoLocal', 
    #                          alternate=True, backend = backend, 
    #                          plot=True, max_bond_dim=None)

if __name__ == '__main__':
    main()