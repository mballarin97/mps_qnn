# This code is part of qcircha.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from qcircha import entanglement_scaling

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
    # backend = 'Aer'
    backend = 'MPS'

    # ------------------------------------------------------------------
    # 1. TOTAL ENTANGLEMENT SCALING
    # If int: test qubits n = 4 to max_num_qubits; If list/array: tests n qubits in list
    max_num_qubits = 6
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