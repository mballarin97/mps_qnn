# This code is part of qcircha.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Plotting libraries and setting
from qcircha import compute_bond_entanglement
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import seaborn as sns
cmap = sns.color_palette('deep', as_cmap=True)

# Useful libraries
import numpy as np
import os # tools per input/outputs
import pprint
pp = pprint.PrettyPrinter(indent=4)

sig = 0.175

# it is called sigma but it is reffered to mean
sigma = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
range_sigma = np.arange(0, len(sigma), dtype=int)

#Load data from np.load
#ent_data=np.empty((15, 7, len(sigma)))
ent_list = []

OUT_PATH = './ent_mean/'
for indx in range_sigma:
    FILE_PATH = os.path.join(OUT_PATH, f'8_mean_{sigma[indx]}_sigma_{sig}.npy')     
    load = np.loadtxt(FILE_PATH)
    ent_list.append(load)

ent_data = np.asarray(ent_list)

ent_haar = ent_data[0, 0, :]
max_haar = ent_haar.max()

ent_max_list = []
for ind in range_sigma:
    max = (ent_data[ind, 7, :]).max()
    print(max)
    ent_max_list.append((max_haar - max))

ent_max = np.asarray(ent_max_list) 
print("ent_max:")
print(ent_max) #blocco:riga:colonna

print("mean:")
print(len(sigma))
print("max:")
print(len(ent_max))

sum_up = np.vstack((sigma, ent_max)).T
print("sum_up:")
print(sum_up)

FILE_PATH = os.path.join(OUT_PATH, f'max_difference_mean_sigma_{sig}.npy') 
np.savetxt(FILE_PATH, sum_up)

#Plot
fig = plt.figure(figsize=(9.6, 6))

plt.plot(sigma, ent_max, ls="--", marker="o", color=cmap[0], label = "Difference values")

plt.legend(loc=1) 
plt.xlabel("Mean")
plt.ylabel("Entanglement entropy")
plt.title(f'Difference between the entanglement entropy maximum')
fig1=plt.gcf()
plt.show()
fig1.savefig(os.path.join(OUT_PATH, f'max_difference_mean_sigma_{sig}.pdf'), format="pdf")



























