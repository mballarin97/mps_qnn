# Entanglement Characterization  
 
 
This folder contains script to study the entanlement in the states created by various PQC, studyied through the lens of MPS accessible quantity, i.e. entanglement entropy.  

---

#### Description of the files
1. `entaglement_characterization.py`: is the main script in the folder, where all the computation happens, and that is imported in all other scripts. Here you can pass a PQC of your choice, and select a simulation backend, MPS or Qiskit's Aer. A number of random parameter vector (100 by default) is generated and the circuit is run this many times, and the entanglement entropy of the final state saved. In addition, there are also script for the evaluation of the entanglement entropy of haar-distributed quantum states. 

2. `var_reps.py`: contains function definitions for multiple analysis (alternate vs. non alternate, varying number of reps, entanglement saturation to haar states) and plots. 

3. `scaling_reps.py`: pretty useless script to evaluate the optimal number of repetition to reach a final entanglement simular to haar-distributed states. Roughly, if the entanglement map is linear, then reps = num_qubits. If entanglement is either a2a or circular, then reps = 0.5 * num_qubits.

4. `circuit.py`: definition of PQC for pre-defined ansatzed or general qnn given a feature map and a variational block. 

5. `visualizer.ipynb`: notebook for data analysys and plotting.

> **Saving data**  
Remember to modify the path were scripts save data! In particular, modify the script `var_reps.py` and be sure that the folder "*data*" (or whatever you called it) exists, as the results are saved only at the end of the whole execution!



