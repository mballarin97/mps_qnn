import numpy as np
from qcircha.expressivity import compute_espressivity
from qcircha.circuit_selector import pick_circuit

def main():

    # SELECT STRUCTURE
    num_qubits = 4
    alternate = True
    repetitions = 7

    # SELECT CIRCUIT 
    feature_map = 'identity'
    var_ansatz = 'circuit10'
    circ = pick_circuit(num_qubits, 1, feature_map=feature_map, var_ansatz=var_ansatz, alternate=alternate)
    circ.draw()

    # OTHER INFO
    backend = 'Aer'
    path = './data/expr/'

    expr = compute_espressivity(num_qubits, repetitions, feature_map = feature_map, var_ansatz=var_ansatz, backend=backend,
                         path=path, plot=True, save=True)

    return expr
                
if __name__ == '__main__':
    
    seed = 120
    np.random.seed(seed)
    
    main()
