import numpy as np
from qcircha.expressivity import compute_espressivity
from qiskit.circuit.library import TwoLocal

def main():

    # SELECT STRUCTURE
    num_qubits = 4
    alternate = True

    # SELECT CIRCUIT 
    feature_map = 'ZZFeatureMap'
    var_ansatz = 'TwoLocal'
    # alternatively, can use a custom parametrized circuit of choice, i.e.
    # feature_map = TwoLocal(..)
    # var_ansatz = QuantumCircuit(...)

    # OTHER INFO
    backend = 'Aer'
    path = './data/expr/'

    repetitions = int(1.5*num_qubits) # test layers L = 1, ..., repetitions
    expr = compute_espressivity(num_qubits, repetitions, feature_map = feature_map, var_ansatz=var_ansatz, 
                                backend=backend, path=path, plot=False, save=True)

    return expr
                
if __name__ == '__main__':
    
    #seed = 120
    #np.random.seed(seed)
    main()
