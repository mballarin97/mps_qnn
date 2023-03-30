[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# Quantum Circuit characterization

Code repository accompanying the paper *[MPS characterization of QNNs, arXiv:2206.02474](https://arxiv.org/abs/2206.02474)*, by Marco Ballarin, Riccardo Mengoni, Stefano Mangini, Chiara Macchiavello and Simone Montangero.

Quantum circuit characterization (`qcircha`) contains the code necessary to characterize the properties of variational quantum circuits, in particular:

- *entanglement*: measured in terms of the *entanglement entropy* across bipartitions of the state created by the parametrized circuit;
- *expressibility*: as introduced in [Sim et al. 2019](https://arxiv.org/abs/1905.10876), is measured as the KL-divergence of the fidelity probability distribution of output states, compared to states sampled according to the Haar distribution.

Both features are computed using an exact simulation of the quantum circuits, leveraging Qiskit's Aer for systems composed of a small number of qubits (tested up to 14 qubits), and a custom MPS simulator for a larger number of qubits (tested up to 50 qubits). Thus, this library enables the user to characterize variational quantum circuits of sizes typical of the NISQ era.

## Installation

To install the library, once the dependencies are installed, simply sun `pip install .` in this directory.

## Usage and files description

The `examples` folder contains the most important scripts used to generate the plots in the manuscript. These and the accompanying notebooks are intended for direct use, while scripts in the `qcircha` directory contain the driving code for the simulations. [Qiskit](https://github.com/Qiskit) is used for the creation and manipulation of Quantum Circuits.

### Examples

To run a simple experiment is is sufficient to run:

```python
from qcircha import entanglement_scaling

alternate = True             # Alternate structure
feature_map = 'ZZFeatureMap' # Feature map of the QNN
var_ansatz = 'TwoLocal'      # Variational ansatz of the QNN
backend = 'Aer'              # Use qiskit backend
max_num_qubits = 6           # Simulation will be ran from 4 to max_num_qubits
OUT_PATH = './ent_scaling/'  # Path to save results
num_trials = 100             # Number of experiments ran for computing the average

entanglement_scaling(max_num_qubits,
                         feature_map=feature_map, var_ansatz=var_ansatz,
                         alternate = alternate, backend = backend, path=OUT_PATH,
                         num_trials = num_trials)
```

The results will be saved inside the `OUT_PATH/` folder, and it will be possible to
access them later. For an example of how to load the data, see the first cells of
the notebook `examples/Entanglement.ipynb`.

However, in the folder `examples` we report the script and notebooks to perform the simulations, analyze the data, and plot the results presented in the paper. The files are:

<details>
   <summary>`ent_study.py`</summary>

   In this example, we show how to study the entanglement production inside a layered QNN with data reuploading with user-defined feature map and variational form. It is possible to use pre-defined circuit templates (see script `circuits.py` and `circuit_selector.py` below for a list of available pre-defined circuits), or even custom parametrized circuits created with Qiskit (in order to work, the circuits must have the attribute `.parameters`). The script can be used to generate data for studying the total entanglement production (function `ent_scaling`) or the entanglement distribution across bonds (`compute_bond_entanglement`).
</details>

<details>
   <summary>`Entanglement.ipynb`</summary>

   Notebook used to analyze and plot the data generated with the `ent_study.py` script.
</details>

<details>
   <summary>`expr_study.py`</summary>

   Used to study the expressibility of a layered QNN with data reuploading with user-defined feature map and variational forms (see above for details on the definition of the circuits).
</details>

<details>
   <summary>`ent_study.py`</summary>
</details>

<details>
   <summary>`gaussian_distribution.py`</summary>

   Example to show how to change the random distribution from which the parameters are sampled.
   This script produces output files that are slightly different from the usual ones, and are
   described in the header of the example file.
</details>

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
- qtealeaves *(only needed for MPS simulation)* [available here](https://baltig.infn.it/quantum_tea_leaves/py_api_quantum_tea_leaves)
- qmatchatea *(only needed for MPS simulation)* [available here](https://baltig.infn.it/quantum_matcha_tea/py_api_quantum_matcha_tea)


## License

The project `qcircha` from the repository https://github.com/mballarin97/mps_qnn
is licensed under the following license:

[Apache License 2.0](LICENSE)

The license applies to the files of this project as indicated
in the header of each file, but not its dependencies.
