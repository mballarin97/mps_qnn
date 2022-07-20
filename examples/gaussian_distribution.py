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
set of qubits, using the gaussian distribution for the random parameters.

The generated files have the following structure:
- The first line is the Von Neumann entropy of a Haar-distributed state
- From line 2 to num_depths+1 the average entropy across the different
  random-sampled parameters
- From num_depths+2 to 2*num_depths+3 the standard deviation of the
  entropies for the different depths of the circuit
"""

from qcircha import compute_bond_entanglement
import numpy as np
from functools import partial
import os

if __name__ == '__main__':
    # Create output directory
    OUT_PATH = 'ent_scaling/'
    if not os.path.isdir(OUT_PATH):
        os.mkdir(OUT_PATH)

    # SELECT PARAMETERS FOR THE GAUSSIAN
    mean = 0
    sigma = 0.1 * np.pi
    distribution = partial(np.random.normal, loc=mean, scale=sigma)

    # Name of the circuit used in saving the files
    circ_name = "gaus"

    # Number of qubits and alternate possibilities
    num_qubits = np.arange(4, 6, 2)
    alternate_possibilities = (True,)

    for num_qub in num_qubits:
        # Depths for the circuit
        max_depth = num_qub
        depths = np.arange(1, int(max_depth), dtype=int)
        for alternate in alternate_possibilities:
            # All computations are done here
            ent, _ = compute_bond_entanglement(num_qub, feature_map='twolocal',
            var_ansatz='twolocal', alternate=alternate, backend='Aer',
            depths=depths, distribution=distribution)

            # Rearranging of the data for easy visualization
            ent_haar = ent[0, 2, :]
            ent_means = ent[:, 0, :]
            ent_std = ent[:, 1, :]

            ent_reorganized = np.vstack( (ent_haar, ent_means, ent_std) )

            # Saving the file
            if alternate:
                FILE_PATH = os.path.join(OUT_PATH, f'alternate_{circ_name}_{num_qub}.npy')
                np.savetxt(FILE_PATH, ent_reorganized)
            else:
                FILE_PATH = os.path.join(OUT_PATH, f'non_alternate_{circ_name}_{num_qub}.npy')
                np.savetxt(FILE_PATH, ent_reorganized)
