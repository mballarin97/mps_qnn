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
Here we generate the data for Fig 8 where we show the marchenko pastur distribution
of the singular value
"""

import numpy as np
from qcircha import pick_circuit
from qmatchatea.py_emulator import QcMps
from qmatchatea import QCConvergenceParameters
from tqdm import tqdm

num_qubits = 15
layers = np.arange(1, num_qubits+5)

fmap = "twolocal"
varans = "twolocal"

singvals_tot = []
for num_layers in tqdm(layers):

    for _ in range(500):
        qc = pick_circuit(num_qubits, num_layers, fmap, varans).decompose()

        num_params = qc.num_parameters
        qc = qc.bind_parameters(
            np.random.uniform(0, np.pi, num_params)
        )

        mps = QcMps(num_qubits, QCConvergenceParameters(int(2**(num_qubits//2)), cut_ratio=1e-20, singval_mode="C"))
        mps.run_from_qk(qc)
        singvals = mps.singvals[num_qubits//2]
        singvals /= np.sqrt( np.sum(singvals**2) )
        singvals_tot.append(singvals)

    np.save(f"data/marchenko-pastur/singvals_c2c2_{num_qubits}_{num_layers}.npy", mps.singvals, allow_pickle=True)



