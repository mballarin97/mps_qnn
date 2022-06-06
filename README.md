[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# Quantum Circuit characterization

Code repository accompanying the paper *[MPS characterization of QNNs, arXiv:XXX](add_link)*, by Marco Ballarin, Riccardo Mengoni, Stefano Mangini, Chiara Macchiavello and Simone Montangero.

Quantum circuit characterization (`qcircha`) contains the code necessary to characterize the properties of variational quantum circuits, in particular:

- *entanglement*: measured in terms of the _entanglement entropy_ across bipartitions of the state created by the parametrized circuit;
- *expressibility*: as introduced in [Sim et al. 2019](https://arxiv.org/abs/1905.10876), is measured as the KL-divergence of the fidelity probability distribution of output states, compared to states sampled according to the Haar distribution.

Both features are computed using an exact simulation of the quantum circuits, leveraging Qiskit's Aer for systems composed of a small number of qubits (tested up to 14 qubits), and a custom MPS simulator for a larger number of qubits (tested up to 50 qubits). Thus, this library enables the user to characterize variational quantum circuits of sizes typical of the NISQ era.

## Installation

To install the library, once the dependencies are installed, simply sun `pip install .` in this directory.

## Usage and files description

The `example` folder contains the most important scripts used to generate the plots in the manuscript. These and the accompanying notebooks are intended for direct use, while scripts in the `qcircha` directory contain the driving code for the simulations. [Qiskit](https://github.com/Qiskit) is used for the creation and manipulation of Quantum Circuits.

### examples/

Here are the script and notebooks to perform the simulations, analyze the data, and plot the results presented in the paper. The files are:

1. `ent_study.py`: used to study the entanglement production inside a layered QNN with data reuploading with user-defined feature map and variational form. It is possible to use pre-defined circuit templates (see script `circuits.py` and `circuit_selector.py` below for a list of available pre-defined circuits), or even custom parametrized circuits created with Qiskit (in order to work, the circuits must have the attribute `.parameters`). The script can be used to generate data for studying the total entanglement production (function `ent_scaling`) or the entanglement distribution across bonds (`compute_bond_entanglement`). 

2. `Entanglement.ipynb`: notebook used to analyze and plot the data generated with the `ent_study.py` script. 

3. `expr_study.py`: used to study the expressibility of a layered QNN with data reuploading with user-defined feature map and variational forms (see above for details on the definition of the circuits).

4. `Entanglement.ipynb`: notebook used to analyze and plot the data generated with the `expr_study.py` script. 

### qcircha

1. `entanglement_characterization.py`: is the main script in the library, where all the computation happens, and that is imported in all other scripts. Here you can pass a PQC of your choice, and select a simulation backend, MPS, or Qiskit's Aer. Several random parameter vectors (100 by default) are generated and the circuit is run this many times, and the entanglement entropy of the final state is saved. In the subdirectiroy `entanglement` there are scripts for the evaluation of the entanglement entropy of quantum states. 

2. `experiments.py`: uses the simulation results from `entanglement_characterization.py` to perform various analyses and plots of the entanglement in the QNN circuit. In particular, here is the code for studying the total entanglement production and the entanglement distribution across bonds.

3. `expressivity.py`: uses the simulation results from `entanglement_characterization.py` to evaluate the expressibility of a QNN, using the definition in [Sim et al. 2019](https://arxiv.org/abs/1905.10876). Such measure requires to construct a histogram of fidelities of states generated by the QNN, to be compared with random states sampled from the uniform Haar distribution. The default number of bins of the histogram is 100, the number of fidelities used to build the histogram is 4950 ( = (100**2 - 100) / 2), obtained by all possible different combinations of the 100 states generated by `entanglement_characterization.py`

4. `circuit.py`: contains some pre-defined parametrized quantum circuits to be used as feature maps or variational forms inside a QNN. Also, a code for creating a general QNN with data reuploading given a feature map, a variational block, and number of layers is present, see function `general_qnn`.

5. `circuit_selector.py`: list of available pre-defined circuits, is used as an intermediate step to create QNNs using the definitions in the script `circuit.py`. Circuits for the `ZZFeatureMap` and `TwoLocal` schemes with all possible entangling topologies are defined.

#### Managing QNNs and simulation results

The scripts `circuits.py` and `circuit_selector.py` contain a list of predefined quantum circuits to be used as feature maps or variational blocks.

All of these are to be used inside the function `general_qnn` which takes a template of a feature map and a variational block and creates the quantum neural network, given a number of repetitions, and order of operations (alternate or sequential).  

All the circuits and simulations results come with metadata information in accompanying `.json` files, specifying the entanglement map (i.e linear/nearest neighbors, ring/circular, full/all to all), as well as the name of the ansatz, and other relevant data used for logging (read below).

#### Saving data

Remember to modify the path where scripts save data! By default simulation results from scripts `ent_study.py` and `expr_study.py` are saved inside the `examples/data/` folder.

Specifically, the result of the executions are saved in:

1. `data/ent_scaling/` for the script `ent_study.py`;
2. `data/expr/` for the script `expr_study.py`.

All data are saved with a unique name given by time of execution followed by a random number, in a `.npy` format. In addition, with the very same name, there is an accompanying `.json` file with information about the performed simulation (i.e. ansatz, entanglement_map,  parameters, ...).

## Requirements

The following packages are required to run the code:

- numpy
- scipy
- matplolib
- qiskit
- tn_py_frontend _(only needed for MPS simulation)_
- qmatchatea _(only needed for MPS simulation)_

The latter two packages are available from Marco Ballarin upon reasonable request.

## License

The project `qcircha` from the repository https://github.com/mballarin97/mps_qnn
is licensed under the following license:

[Apache License 2.0](LICENSE)

The license applies to the files of this project as indicated
in the header of each file, but not its dependencies.
