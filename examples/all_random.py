# This code is part of qcircha.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""
Here we generate the data for Fig 7 where ALL the parameters are random
without any data reuploading
"""
import os
import numpy as np
from qcircha.circuit_selector import _select_circ
from qiskit import QuantumCircuit
import qmatchatea as qmt
from qmatchatea.qk_utils import qk_transpilation_params
import qtealeaves.observables as obs
from qtealeaves.emulator import MPS
from tqdm import tqdm

depth = 14
dir_name = f"data/all_random"
num_qubits = 13
num_reps = 1
num_trials = 1
max_bond_dim = 128

if not os.path.isdir(dir_name):
    os.makedirs(dir_name)

observables = obs.TNObservables()
observables += obs.TNObsBondEntropy()
observables += obs.TNState2File('mps_state.txt', 'F')
conv_params = qmt.QCConvergenceParameters(int(max_bond_dim), singval_mode="C")
backend = qmt.QCBackend()
trans_params = qk_transpilation_params(linearize=True, tensor_compiler=False)

feature_map = _select_circ(num_qubits, "zzfeaturemap")
ansatz = _select_circ(num_qubits, "twolocal")
num_avg = 1000

for idx, data in tqdm(enumerate(range(num_avg))):
    entanglement_total = []
    singvals_cut_total = []
    states_total = []
    initial_state = "Vacuum"
    for num_reps in range(depth):
        print(f"\n__Reps {num_reps}/{depth}")
        random_params = np.pi * np.random.rand(len(ansatz.parameters))
        random_params_data = np.pi * np.random.rand(len(feature_map.parameters))

        binded_ansatz = ansatz.bind_parameters(random_params)
        binded_fmap = feature_map.bind_parameters(random_params_data)

        # Pick a PQC (modify the function)
        qc = QuantumCircuit(num_qubits)
        qc = qc.compose(binded_fmap)
        qc = qc.compose(binded_ansatz)

        io_info = qmt.QCIO(initial_state=initial_state)
        results = qmt.run_simulation(qc, convergence_parameters=conv_params,
                                transpilation_parameters=trans_params, backend=backend,
                                observables=observables, io_info=io_info)

        entanglement = np.array(list(results.entanglement.values()))
        initial_state = results.mps

        entanglement_total.append(entanglement)
        states_total.append(results.singular_values_cut)
        initial_state = MPS.from_tensor_list(initial_state)

        np.save( os.path.join(dir_name, f"entanglement_{idx+num_avg}.npy"), entanglement_total, allow_pickle=True)
