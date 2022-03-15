import numpy as np
from qcircha import entanglement_scaling, compute_bond_entanglement
from qiskit.circuit.library import TwoLocal

def main():

    # SELECT STRUCTURE
    alternate = True

    # SELECT CIRCUIT
    feature_map = 'ZZFeatureMap'
    var_ansatz = 'TwoLocal'
    # alternatively, can use a custom parametrized circuit of choice, i.e.
    # feature_map = TwoLocal(..)
    # var_ansatz = QuantumCircuit(...)

    # SIMULATION BACKEND
    backend = 'Aer'
    # backend = 'MPS'

    # ------------------------------------------------------------------
    # 1. TOTAL ENTANGLEMENT SCALING    
    # If int: test qubits n = 4 to max_num_qubits; If list/array: tests n qubits in list
    max_num_qubits = 12                      
    entanglement_scaling(max_num_qubits, 
                         feature_map=feature_map, var_ansatz=var_ansatz,
                         alternate = alternate, backend = backend,
                         max_bond_dim = 1024, path='./data/ent_scaling/')

    # ------------------------------------------------------------------
    # 2. ENTANGLEMENT ACROSS BIPARTITIONS
    # num_qubits = 5
    # compute_bond_entanglement(num_qubits, 
    #                          feature_map='ZZFeatureMap', var_ansatz='TwoLocal', 
    #                          alternate=True, backend = backend, 
    #                          plot=True, max_bond_dim=None)

if __name__ == '__main__':

    #seed = 34
    #np.random.seed(seed)
    main()