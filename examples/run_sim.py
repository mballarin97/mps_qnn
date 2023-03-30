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
Compute and save the bond entanglement of a given architecture for a given
set of qubits
"""

from qcircha import compute_bond_entanglement
import numpy as np

num_qubits = [16]
alternate_possibilities = (True,)
backend = "MPS"

for num_qub in num_qubits:
    for alternate in alternate_possibilities:
        ent, _, times = compute_bond_entanglement(num_qub, feature_map='ZZFeatureMap',
        var_ansatz='twolocal', alternate=alternate, backend=backend, num_trials=100,
        max_bond_dim=2**8)

        ent_haar = ent[0, 2, :]
        ent_means = ent[:, 0, :]
        ent_std = ent[:, 1, :]

        ent_reorganized = np.vstack( (ent_haar, ent_means, ent_std) )

        if alternate:
            np.savetxt(f'data/marcos_plot/alternate_abbas{num_qub}.npy', ent_reorganized)
            np.savetxt(f'data/marcos_plot/alternate_abbas{num_qub}_{backend}_time.npy', times)
        else:
            np.savetxt(f'data/marco_ent_scaling/non_alternate_abbas{num_qub}.npy', ent_reorganized)
