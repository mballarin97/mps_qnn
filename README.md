[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
# Quantum Circuit characterization

Repository with the code for the collaboration between Marco Ballarin, Riccardo Mengoni, Stefano Mangini.

Quantum circuit characterization (`qcircha`) contains the code necessary to characterize a variational quantum circuit
on the following aspects:

- the entanglement scaling
- the KL-divergence of the probability distribution of the outputs compared to the Haar distribution

Both the aspects are computed for an exact simulation with a reduced number of qubits using qiskit, and for a larger
number of qubits using an MPS simulator. This library so enable the user to characterize a variational quantum circuit
of almost any size compatible to the NISQ era.

## Usage

#### Description of the files
1. `run_simulations.py`: is the main script in the folder, where all the computation happens, and that is imported in all other scripts. Here you can pass a PQC of your choice, and select a simulation backend, MPS or Qiskit's Aer. A number of random parameter vector (100 by default) is generated and the circuit is run this many times, and the entanglement entropy of the final state saved. In addition, there are also script for the evaluation of the entanglement entropy of haar-distributed quantum states. 

2. `entanglement.py`: contains multiple analysis of the entanglement in the MPS circuit (alternate vs. non alternate, varying number of reps, entanglement saturation to haar states) and plots.

3. `scaling_reps.py`: pretty useless script to evaluate the optimal number of repetition to reach a final entanglement simular to haar-distributed states. Roughly, if the entanglement map is linear, then reps = num_qubits. If entanglement is either a2a or circular, then reps = 0.5 * num_qubits.

4. `circuit.py`: definition of PQC for pre-defined ansatzed or general qnn given a feature map and a variational block. It is much improved wrt to the original file in the parent folder, as it now includes other ansatzes and the definition of names and metadatas for each circuit (see below).

5. `visualizer.ipynb`: notebook for data analysys and plotting.  

---  


#### Creating a QNN
The script `circuits.py` contains a list of predefined quantum circuits to be used as feature maps or variational blocks. 

All of these are to be used inside the function `general_qnn()` which takes a template of a feature map and a variational block and creates the quantum neural network, given a number of repetitions, and order of operations (alternate or not).  

All the circuits come with metadata information specifying the entanglement map (i.e linear, ring, a2a), as well as the name of the ansatz, and other relevant data used for logging.  

---  


#### Saving data
> **MODIFY PATH**  
Remember to modify the path were scripts save data! In particular, modify the script `var_reps.py` and be sure that the folder "*data*" (or whatever you called it) exists, as the results are saved only at the end of the whole execution!

The result of the executions are saved in a folder named "data", and in the subfolders:
1. `data/ent_scaling/` for the script `entanglement.py` (see the funtion `entanglement_scaling`);
2. `data/optimal_reps/` for the script `scaling_reps.py`.

All data are saved with a unique name given by time of execution followed by a random indx, in a `.npy` format. In addition, with the same name there is an accompanying `.json` file wiht information about the performed simulation (i.e. ansatz, entanglement_map,  parameters, ...).


## Requirement

To properly run these code you need the following packages:

- numpy
- qiskit
- tn_py_frontend
- qcomps

The latter two packages are available from Marco Ballarin upon reasonable request

## License

The project `qcircha` from the repository https://github.com/mballarin97/mps_qnn
is licensed under the following license:

[Apache License 2.0](LICENSE)

The license applies to the files of this project as indicated
in the header of each file, but not its dependencies.