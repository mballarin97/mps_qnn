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

from qcircha import compute_bond_entanglement # computa l'entanglement delle diverse bipartizioni 
import numpy as np
from functools import partial 
import os # tools per input/outputs

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import seaborn as sns
cmap = sns.color_palette('deep', as_cmap=True)

if __name__ == '__main__': 
    # Create output directory

    OUT_PATH = 'ent_sigma/' 
    if not os.path.isdir(OUT_PATH):
        os.mkdir(OUT_PATH)

    # SELECT PARAMETERS FOR THE GAUSSIAN
    mean = np.pi/2.
    sigma = 10. * np.pi

    distribution = lambda x : np.random.normal(mean, sigma, x)
    # Definisco una nuova funzione dove SOLO lambdax è la variabile
    # Prendo in considerazione una distribuzione gaussiana e fisso media e sigma di essa.
    # L'unico parametro della distribuzione che rimane è il numero di parametri 
    # che voglio estrarre da tale distribuzione per inserirli nel circuito analizzato

    # Name of the circuit used in saving the files
    # Nome del circuito in base alla sigma scelta
    circ_name = f'sigma_{sigma/np.pi}'

    # Number of qubits and alternate possibilities
    num_qubits = np.arange(8, 10, 2) 
    print("NUM QBITS:")
    print(num_qubits)
    #vettore che tiene in memoria il range di num_qubit selezionato, in questo caso è solo 4
    alternate_possibilities = (True,) 
    #genera blocchi di F e V alternati tra loro

    #ciclo for che mi analizza i circuiti con un numero di qbit preso dal range e sempre diverso
    for num_qub in num_qubits: 
        # Depths for the circuit
        max_depth = num_qub 
        #massimo numero di blocchi del circuito (un blocco di F e V)
        #perchè è stato osservato che la depth pari al numero di qbit mi porta alla saturazione dell'entanglement
        depths = np.arange(1, int(max_depth), dtype=int) 
        #salvo il range di profondità in un vettore
        for alternate in alternate_possibilities: 
            # All computations are done here
            ent, _ = compute_bond_entanglement(num_qub, feature_map='twolocal',
            var_ansatz='twolocal', alternate=alternate, backend='Aer',
            depths=depths, distribution=distribution) 

            #print("PRINT DI PROVA:")
            #print(ent)
            #,_ sono gli output della funzione che non ci interessano, l'unico output interessante è la misura di entanglement

            # Rearranging of the data for easy visualization
            ent_haar = ent[0, 2, :] #[blocco, riga, colonna] e ":" = prendi tutto
            #print("PRINT HAAR:")
            #print(ent_haar)
            #upperbound, massimo entanglement ottenibile
            ent_means = ent[:, 0, :] 
            print("PRINT MEANS:")
            print(ent_means)
            #media entanglement calcolato per i diversi set di parametri ricavati dalla distribuzione
            ent_std = ent[:, 1, :] 
            #print("PRINT STD:")
            #print(ent_std)
            #std della media dell'entamnglement sui vari set di parametri ricavati dalla distribuzione

            ent_reorganized = np.vstack( (ent_haar, ent_means, ent_std) ) 
            #print("PRINT ENT_REO")
            #print(ent_reorganized)

            # Saving the file
            #non ho ben capito come funziona e come posso leggere da file i dati .npy
            if alternate:
                FILE_PATH = os.path.join(OUT_PATH, f'alternate_{circ_name}_{num_qub}.npy') 
                np.savetxt(FILE_PATH, ent_reorganized) 
            else:
                FILE_PATH = os.path.join(OUT_PATH, f'non_alternate_{circ_name}_{num_qub}.npy')
                np.savetxt(FILE_PATH, ent_reorganized)

            #avrò alla fine n-1 valori di entanglement se lavoro con n qbit
            #In questo caso ne ho 3 perchè lavoro con 4 qbit

    #Plot
    x = np.arange(1, num_qubits, 1)
    #DEVO CAPIRE COME FARMI PLOTTARE SOLO GLI INTERI DULLE X
    fig = plt.figure(figsize=(9.6, 6))
    #4 QBITS:
    #plt.plot(x, ent_haar, '--ko', x, ent_means[0, :], '-bo', x, ent_means[1, :],'-go', x, ent_means[2, :], '-ro')
    #legend=plt.legend(['Haar random','L=1', 'L=2', 'L=3'])

    #8QBITS:

    range = np.arange(0, len(ent_means), dtype=int)
    for indx in range:
        plt.plot(x, ent_means[indx, :], ls="-",
        marker="o", c=cmap[indx], label=f'L=_{indx+1}')

    plt.plot(x, ent_haar, ls="--",
    marker="o", color=cmap[indx+1], label= 'Haar-random')  

    plt.legend(loc=4)    
    plt.xlabel("Bond index cut ")
    plt.ylabel("Entanglement entropy")
    plt.title(f'Entanglement distribution across bonds_{circ_name}')
    fig1=plt.gcf()
    plt.show()
    fig1.savefig(os.path.join(OUT_PATH, f'alternate_{circ_name}_{num_qub}.pdf'), format="pdf")
           
