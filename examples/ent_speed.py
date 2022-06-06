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
Compute the entanglement speed of a given QNN
"""

from qcircha import compute_bond_entanglement
from qcircha.entanglement import haar_bond_entanglement
import numpy as np
from scipy.stats import linregress

#('zzfeaturemap', 'twolocal')
circuits = (('circuit1', 'twolocal'), ('zzfeaturemap', 'twolocal_parametric2q'), ('circuit1', 'twolocal_parametric2q') )

num_qubits = np.arange(8, 17, 2)

def obtain_xy(circ_name, num_qubs):
    abbas = {nq:[ np.loadtxt(circ_name + f'{nq}.npy'), 0] for nq in num_qubs}

    xx = []
    yy = []
    for idx, qub in enumerate(num_qubs):
        data = abbas[qub][0]
        num_reps = min(int( (len(data)-1 )/2 )+1, 20)
        reps = np.arange(1, num_reps)

        yval = data[1:num_reps, :].max(axis=1)/max(haar_bond_entanglement(qub))

        xx += list( reps/qub )
        yy += list(yval)

    xx = np.array(xx)
    yy = np.array(yy)

    return xx, yy

def fit(xx, yy):
    mask = xx < 0.5
    yy1 = yy[mask]
    xx1 = xx[mask]
    xx1 = np.log(xx1)
    yy1 = np.log(yy1)
    res = linregress(xx1, yy1)

    return res.slope, res.stderr

with open('final_res.txt', 'w') as fh:
    fh.write('Feature map \t Variational ansatz \t Entangling speed \n')
    for circ in circuits:
        fmap = circ[0]
        vmap = circ[1]

        filename = f'collapsing/{fmap}_{vmap}_'
        for num_qub in num_qubits:
            ent, _ = compute_bond_entanglement(num_qub, feature_map=fmap,
            var_ansatz=vmap, alternate=True, backend='Aer')

            ent_haar = ent[0, 2, :]
            ent_means = ent[:, 0, :]
            ent_std = ent[:, 1, :]

            ent_reorganized = np.vstack( (ent_haar, ent_means, ent_std) )

            np.savetxt(filename + f'{num_qub}.npy', ent_reorganized)

        xx, yy = obtain_xy(filename, num_qubits)
        vs, vs_err = fit(xx, yy)
        fh.write(f'{fmap} \t {vmap} \t ({vs}+-{vs_err}) \n')


