import numpy as np
from qcircha.expressivity import compute_espressivity

def main():
    # FIX SEED
    seed = 120
    np.random.seed(seed)

    # SELECT HYPER-PARAMETERS
    num_qubits = 4
    alternate = True
    repetitions = 10

    backend = 'Aer'
    path = './data/expr/'

    # SELECT CIRCUIT 
    feature_map = 'identity'
    var_ansatz = 'circuit15'
    # ZZFeatureMap(num_qubits, reps=1, entanglement='linear')
    # TwoLocal(num_qubits, 'ry', 'cx', 'linear', reps=1, insert_barriers=True, skip_final_rotation_layer=True)
    # qk.QuantumCircuit(num_qubits, name="Id", metadata={'entanglement_map': None}) # Identiy

    compute_espressivity(num_qubits, repetitions, feature_map = feature_map, var_ansatz=var_ansatz, backend=backend,
                         path=path, plot=True, save=True)
                
if __name__ == '__main__':
    main()
