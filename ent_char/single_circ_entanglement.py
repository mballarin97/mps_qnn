from qcomps.interface import run_simulation
from qcomps.utils import QCConvergenceParameters
from qcomps.qk_utils import qk_transpilation_params
import numpy as np
import os
from circuits import piramidal_circuit

# Number of simulation over which we average
num_avg = 20
# Maximum bond dimension achievable
max_bond_dim = 1024
# PATH to the results folder
res_PATH = 'results/'
# Set seed
np.random.seed(47)

# Create output folder if that does not exist
if not os.path.isdir(res_PATH):
    os.mkdir(res_PATH)


# Number of qubits searched
num_qub_range = [4, 10, 20, 30, 50]
# Number of repetitions checked
reps = np.arange(1, 6, 2)

entanglement_avg = np.zeros( len(reps) )
entanglement_std = np.zeros( len(reps) )
bond_dim_avg = np.zeros( len(reps) )
bond_dim_std = np.zeros( len(reps) )

# ==== Cycle over number of qubits ====
for num_qubits in num_qub_range:

    # ==== Cycle over the different circuit repetitions ====
    for idx, rep in enumerate(reps):
        avg_entanglement_temp = np.zeros(num_avg)
        max_bond_dim_achieved_temp = np.zeros(num_avg)

        # ==== Cycle over averages ====
        for avg in range(num_avg):
            # Create circuit (Substitute with the function for creating the circuit you like)
            circuit = piramidal_circuit(num_qubits, rep)

            # ======= Draw circuit parameters, uniformly in (0,2pi)
            params = np.random.uniform(0, 2*np.pi, len(circuit.parameters) )
            params = { list(circuit.parameters)[ii] : params[ii] for ii in range(len(params))}

            # ======= Bind parameter values and construct circuit
            circuit = circuit.bind_parameters(params)

            # ======= Simulation parameters
            conv_params = QCConvergenceParameters(max_bond_dim)
            # Put linearize to True if your circuit is not a Nearest Neighbors architecture
            trans_params = qk_transpilation_params(linearize = False)

            # Run the simulation with MPS. See simulation_results in qcomps for the attributes of the
            # results parameters
            results = run_simulation(circuit, convergence_parameters=conv_params,
                transpilation_parameters=trans_params, do_entanglement=True, approach='PY',
                save_mps='F')

            max_bond_dim_achieved_temp[avg] = np.max( [ np.max(tens.shape ) for tens in results.mps] )
            avg_entanglement_temp[avg] = np.max(results.entanglement)

        entanglement_avg[idx] = avg_entanglement_temp.mean()
        entanglement_std[idx] = avg_entanglement_temp.std()
        bond_dim_avg[idx] = max_bond_dim_achieved_temp.mean()
        bond_dim_std[idx] = max_bond_dim_achieved_temp.std()

    np.savetxt(os.path.join(res_PATH, f'entantglement_{num_qubits}qubits_avg.npy'), entanglement_avg)
    np.savetxt(os.path.join(res_PATH, f'entantglement_{num_qubits}qubits_std.npy'), entanglement_std)
    np.savetxt(os.path.join(res_PATH, f'bonddim_{num_qubits}qubits_avg.npy'), bond_dim_avg)
    np.savetxt(os.path.join(res_PATH, f'bonddim_{num_qubits}qubits_std.npy'), bond_dim_std)

print('===================== FINISHED =====================')