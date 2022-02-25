import numpy as np
from qcircha import entanglement_scaling

def main():
    seed = 42
    np.random.seed(seed)

    # Quantum Cirucit structure
    #num_qubits = 8
    alternate = True

    # Choose simulation backend
    backend = 'MPS'
    #backend = 'Aer'

    max_num_qubits = 4 #np.arange(30, 51, 10)
    entanglement_scaling(max_num_qubits, 
                        feature_map = 'Circuit9', var_ansatz = "TwoLocal", 
                        alternate = alternate, backend = backend,
                        max_bond_dim = 1024, path='./data/ent_scaling/')

    #main(num_qubits, backend=backend, alternate=alternate)
    #ent_vs_reps(num_qubits, alternate = alternate, backend=backend)
    #alt_comparison(num_qubits, backend=backend)

if __name__ == '__main__':
    main()