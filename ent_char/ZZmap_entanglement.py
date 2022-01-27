from qcomps.interface import run_simulation
from qcomps.utils import QCConvergenceParameters
from qcomps.qk_utils import qk_transpilation_params
import numpy as np
from qiskit.circuit.library import ZZFeatureMap, TwoLocal


num_avg = 20
max_bond_dim = 1000
res_PATH = 'results/'

num_qub_range = [4, 10, 20, 30, 50]
ansatz_reps = np.arange(1, 6, 2)
feature_reps = np.arange(1, 6, 2)
entanglement_avg = np.zeros(( len(feature_reps), len(ansatz_reps)) )
entanglement_std = np.zeros( (len(feature_reps), len(ansatz_reps)) )
bond_dim_avg = np.zeros(( len(feature_reps), len(ansatz_reps)) )
bond_dim_std = np.zeros( (len(feature_reps), len(ansatz_reps)) )


for num_qubits in num_qub_range:
    for a_idx, areps in enumerate(ansatz_reps):
        var_ansatz = TwoLocal(num_qubits, 'ry', 'cx', 'linear', reps=areps)

        for f_idx, freps in enumerate(feature_reps):
            feature_map = ZZFeatureMap(num_qubits, reps=freps, entanglement='linear')

            avg_entanglement_temp = np.zeros(num_avg)
            max_bond_dim_achieved_temp = np.zeros(num_avg)
            for avg in range(num_avg):
                # ======= Draw Feature Map parameter, uniformly in (0,1)
                feature_params = np.random.uniform(0, 1, len(feature_map.parameters))
                # ======= Draw ansatz parameters, uniformly in (0, 2pi)
                ansatz_params = np.random.uniform(0, 2*np.pi, len(var_ansatz.parameters) )

                # ======= Bind parameter values and construct circuit
                binded_fm = feature_map.bind_parameters(feature_params).decompose()
                binded_ansatz = var_ansatz.bind_parameters(ansatz_params).decompose()
                circuit = binded_fm.compose(binded_ansatz)

                # ======= Simulation parameters
                conv_params = QCConvergenceParameters(max_bond_dim)
                trans_params = qk_transpilation_params(linearize = False)

                results = run_simulation(circuit, convergence_parameters=conv_params,
                    transpilation_parameters=trans_params, do_entanglement=True, approach='PY',
                    save_mps='F')

                max_bond_dim_achieved_temp[avg] = np.max( [ np.max(tens.shape ) for tens in results.mps] )
                avg_entanglement_temp[avg] = np.max(results.entanglement)

            entanglement_avg[a_idx, f_idx] = avg_entanglement_temp.mean()
            entanglement_std[a_idx, f_idx] = avg_entanglement_temp.std()
            bond_dim_avg[a_idx, f_idx] = max_bond_dim_achieved_temp.mean()
            bond_dim_std[a_idx, f_idx] = max_bond_dim_achieved_temp.std()

    np.savetxt(res_PATH+f'entantglement_{num_qubits}qubits_avg.npy', entanglement_avg)
    np.savetxt(res_PATH+f'entantglement_{num_qubits}qubits_std.npy', entanglement_std)
    np.savetxt(res_PATH+f'bonddim_{num_qubits}qubits_avg.npy', bond_dim_avg)
    np.savetxt(res_PATH+f'bonddim_{num_qubits}qubits_std.npy', bond_dim_std)

print('===================== FINISHED =====================')