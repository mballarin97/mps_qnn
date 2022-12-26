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

#tools for graphic title
from math import log10, floor
from decimal import Decimal

#FUNZIONE PER SCRITTURA DECIMALE
def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]


#SIGMA VECTOR
sigma = np.logspace(-5, 3, 15)
range_sigma = np.arange(0, len(sigma), dtype=int)
sigma_pi_list = []
for i in sigma:
    mean = np.pi/2.
    sigma_round = round(i, -int(floor(log10(abs(i)))) )
    sigma_pi_list.append(format_e(Decimal(sigma_round)))
sigma_pi = np.asarray(sigma_pi_list)
print(sigma_pi)

ent_max_tot_list = []

# Nmber of qubits and alternate possibilities
num_qubits = np.arange(8, 16, 2) 
print("NUM QBITS:")
print(num_qubits)

for num_qub in num_qubits: 

    PATH = 'ent_sigma/' 
    OUT_PATH = os.path.join(PATH, f'{num_qub}_qubit')
    if not os.path.isdir(OUT_PATH):
        os.mkdir(OUT_PATH)

    #Load data from np.load
    #ent_data=np.empty((15, 7, len(sigma)))
    ent_list = []
    for indx in range_sigma:
        FILE_PATH = os.path.join(OUT_PATH, f'{num_qub}_sigma_{sigma_pi[indx]}.npy')     
        load = np.loadtxt(FILE_PATH)
        ent_list.append(load)

    ent_data = np.asarray(ent_list)
    ent_haar = ent_data[0, 0, :]
    max_haar = ent_haar.max()

    ent_max_list = []
    for ind in range_sigma:
        max = (ent_data[ind, 7, :]).max()
        #print(max)
        ent_max_list.append((max_haar - max))


    ent_max = np.asarray(ent_max_list) 
    ent_max_tot_list.append(ent_max)

    print("ent_max:")
    print(ent_max) #blocco:riga:colonna

ent_max_tot = np.asarray(ent_max_tot_list) 
print("ent_max_tot")
print(ent_max_tot)

sum_up = np.vstack((sigma, ent_max_tot))
print("sum_up:")
print(sum_up)

#FILE_PATH = os.path.join(OUT_PATH, f'max_difference.npy') 
#np.savetxt(FILE_PATH, sum_up)

#Plot
fig = plt.figure(figsize=(9.6, 6))
range = np.arange(0, len(num_qubits), dtype=int)

print("dimensioni")
print(len(sum_up[1, :]))
print(len(sigma))

for indx in range:
    print(indx)
    plt.plot(sigma, sum_up[indx+1, :], ls="-", marker="o", c=cmap[indx], label=f'n={indx*2 + 8}')

plt.xscale('log')
plt.yscale('log')
plt.legend(loc=1) 
plt.xlabel("Variance $\sigma$")
plt.ylabel("Difference between the entanglement entropy maximum")
fig1=plt.gcf()
plt.show()
fig1.savefig(os.path.join(PATH, f'max_difference_loglog.pdf'), format="pdf")



























