"""
Contains main computational methods
"""

# Import necessary modules
import time

from qcomps import run_simulation
from qcomps.qk_utils import qk_transpilation_params
from qcomps import QCConvergenceParameters
from circuits import general_qnn
import numpy as np
import matplotlib.pyplot as plt

from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit.quantum_info.states.utils import partial_trace
from qiskit.quantum_info.states.measures import entropy
from qiskit import Aer, transpile
sim_bknd = Aer.get_backend('statevector_simulator')

def run_mps(qc, max_bond_dim=1024):
    """
    Runs a quantum circuit (parameters already numerical) with MPS.

    Parameters
    ----------
    qc : :py:class:`QuantumCircuit`
        qiskit quantum circuit class
    max_bond_dim: int, optional
        Maximum bond dimension for the MPS. Default to 1024

    Returns
    -------
    :py:class:`qcomps.simulation_results`
        The results of the simulations, comprehensive of the entanglement
    """

    conv_params = QCConvergenceParameters(max_bond_dim)
    trans_params = qk_transpilation_params(linearize=True)

    results = run_simulation(qc, convergence_parameters=conv_params,
                             transpilation_parameters=trans_params, do_entanglement=True,
                             approach='PY', save_mps='N', do_statevector=False)

    return results

def run_circuit(qc, parameters, max_bond_dim=1024):
    """
    Assigns parameters to the quantum circuits and runs it with python-MPS.

    Parameters
    ----------
    qc : :py:class:`QuantumCircuit`
        qiskit quantum circuit class
    parameters : array-like
        Array of parameters, which length should match the number of parameters
        in the variational circuit
    max_bond_dim: int, optional
        Maximum bond dimension for the MPS. Default to 1024.

    Returns
    -------
    :py:class:`qcomps.simulation_results`
        The results of the simulations, comprehensive of the entanglement
    """

    qc = qc.assign_parameters(parameters)
    results = run_mps(qc, max_bond_dim=max_bond_dim)
    return results

def harmonic(n):
    """
    Approximation of the Harmonic series.
    This approximation is really useful to speed up the computation of the
    Haar entanglement of a circuit, which is exponential to compute if computed
    exactly. The error of the approximation is exponentially small, since in
    our case :math:`n\propto 2^{num_qubits}`.
    
    .. math::
        H(n) = \sum_{k=1}^n \frac{1}{k} ~ \ln(n) + \gamma + O(\frac{1}{n} )

    Parameters
    ----------
    n : int
        Truncation of the Harmonic series

    Returns
    -------
    float
        The value of :math:`H(n)`
    """
    return np.log(n) + np.euler_gamma

def approx_haar_entanglement(num_qubits, num_A):
    """
    Approximate expression of the haar_entanglement function,
    very accurate for num_qubits > 10 (error < 1%).
    For a more accurate description check :py:func:`haar_entanglement`

    Parameters
    ----------
    num_qubits : int
        Number of qubits
    num_A : int
        Number of qubits in the partition A

    Return
    ------
    float
        Approximate haar entanglement of the bipartition
    """
    num_A = min([num_A, num_qubits - num_A])
    d = (2**num_A - 1) / (2**(num_qubits - num_A + 1))
    ent = harmonic(2**num_qubits) - harmonic(2**(num_qubits - num_A)) - d
    return ent

def haar_entanglement(num_qubits, num_A, log_base = 'e'):
    """
    Entanglement entropy of a random pure state (haar distributed),
    taken from Commun. Math. Phys. 265, 95–117 (2006), Lemma II.4.
    Considering a system of num_qubits, bi-partition it in system A
    with num_A qubits, and B with the rest. 
    Formula applies for :math:`num_A \leq num_B`.

    .. warning::
        This function computational time scales exponentially with
        the number of qubits. For this reason, if `num_qubits`>25
        the approximate function :py:func`approx_haar_entanglement`
        is instead used

    Parameters
    ----------
    num_qubits : int
        Number of qubits
    num_A : int
        Number of qubits in the partition A
    log_base : str, optional
        Base of the logarithm.
        Possibilities are: 'e', '2'.
        Default is 'e'.

    Return
    ------
    float
        Approximate haar entanglement of the bipartition
    """

    # Pick the smallest bi-partition
    num_A = min([num_A, num_qubits - num_A])

    dim_qubit = 2  # qubit has dimension 2
    da = dim_qubit ** num_A  # dimension of partition A
    db = dim_qubit ** (num_qubits - num_A)  # dimension of partition B
    
    if num_qubits > 25:
        ent = approx_haar_entanglement(num_qubits, da)
    else:
        ent = np.sum([1.0 / j for j in range(1+db, da * db + 1)])
        ent -= (da - 1) / (2*db)

    if log_base == '2': 
        ent *= 1 / np.log(2)

    return ent

def haar_bond_entanglement(num_qubits):
    """
    Evaluates the expected value of the entanglement at each bond if the
    states were Haar distributed. To avoid an exponential scaling of the
    computational time the approximation for the Haar entanglement is used
    for `num_qubits`>20. For a definition of the Haar entanglement see
    :py:func:`haar_entanglement`

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit

    Returns
    -------
    array-like of floats
        Haar entanglement along all `num_qubits-1` bipartitions
    """

    if num_qubits < 20:
        entanglement_bonds = [haar_entanglement(num_qubits, i) for i in range(1, num_qubits)]
    else:
        entanglement_bonds = [approx_haar_entanglement(num_qubits, i) for i in range(1, num_qubits)]
    
    return entanglement_bonds

def entanglement_entropy(statevector, idx_to_trace=None):
    """
    Entanglement entropy of subsystem of a pure state.
    Given a statevector (i.e. pure state), builds the density matrix,
    and traces out some systems.
    Then eveluates Von Neumann entropy using Qiskit's implementation.
    Be consistent with the base of the logarithm.

    Parameters
    ----------
    statevector : array-like
        Statevector of the system
    idx_to_trace : array-like, optional
        Indexes to trace away. By default None.
    
    Returns
    -------
    float
        Entanglement entropy of the reduced density matrix obtained from statevector
        by tracing away the indexes selected
    """
    # Construct density matrix
    rho = np.outer(statevector, np.conjugate(statevector))
    # Trace out part of the system (indicated by idx_to_trace, following Qiskit indexing)
    partial_rho = partial_trace(rho, idx_to_trace)
    ent_entropy = entropy(partial_rho, base = np.e)

    return ent_entropy

def entanglement_bond(statevector):
    """
    Evaluates the entanglement entropies for cuts in a 1-d quantum system (as in MPS), given a pure state 
    statevector of num_qubits qubits, thus with length 2**num_qubits.
    Must be a valid statevector: amplitudes summing to one.

    Parameters
    ----------
    statevector : array-like
        Statevector of the system
    
    Returns
    -------
    array-like of floats
        Entanglement entropy along all the bipartitions of the system
    """
    sum_amplitudes = np.sum(np.vdot(statevector, statevector) )
    if not np.isclose(sum_amplitudes, 1):
        raise ValueError(f'Amplitudes of the statevector must sum up to 1, not to {sum_amplitudes}')

    num_qubits = int(np.log2(len(statevector)))
    
    # Cut along the first half of the bond
    res1 = [entanglement_entropy(statevector, idx_to_trace=range(i+1, num_qubits)) for i in range(int(num_qubits/2))]
    
    # Cut along the second half
    res2 = [entanglement_entropy(statevector, idx_to_trace=range(0, i)) for i in range(1+int(num_qubits/2), num_qubits)]
    res = res1 + res2

    return np.array(res)

def mps_simulation(qc, random_params, max_bond_dim=1024):
    """"
    Simulation using MPS to study bond entanglement.

    Parameters
    ----------
    qc : :py:class:`QuantumCircuit`
        qiskit quantum circuit class
    random_params : array-like of array-like
        Sets of random parameters over which obtain the average
    max_bond_dim : int, optional
        Maximum bond dimension for MPS simulation

    Returns
    -------
    array-like of floats
        Average of the entanglement over different parameter sets
    array-like of floats
        Standard deviation of the entanglement over different parameter sets
    """
    sim_bknd = Aer.get_backend('statevector_simulator')
        
    mps_results_list = []
    for idx, params in enumerate(random_params):
        print(f"Run {idx}/{len(random_params)}", end="\r")
        qc = transpile(qc, sim_bknd) # Why this transpilation?
        mps_results = run_circuit(qc, params, max_bond_dim=max_bond_dim)
        mps_results_list.append(mps_results)

    mps_entanglement = np.array([res.entanglement for res in mps_results_list])
    ent_means = np.mean(mps_entanglement, axis=0)
    ent_std = np.std(mps_entanglement, axis=0)

    return ent_means, ent_std

def aer_simulation(qc, random_params, get_statevector = False):
    """
    Simulation using Qiskit Aer to study bond entanglement.

    TODO: add **kwargs to customize the backend

    Parameters
    ----------
    qc : :py:class:`QuantumCircuit`
        qiskit quantum circuit class
    random_params : array-like of array-like
        Sets of random parameters over which obtain the average

    Returns
    -------
    array-like of floats
        Average of the entanglement over different parameter sets
    array-like of floats
        Standard deviation of the entanglement over different parameter sets
    array-like of floats or None
        statevector of the system
    """

    qk_results_list = []
    sim_bknd = Aer.get_backend('statevector_simulator')
    for idx, params in enumerate(random_params):
        print(f"Run {idx}/{len(random_params)}", end="\r")
        qc_t = qc.decompose()
        qk_results = sim_bknd.run(qc_t.assign_parameters(params))
        qk_results_list.append(np.asarray(qk_results.result().get_statevector()))
    
    print("\nPartial tracing...")
    now = time.time()
    aer_ent = np.array([entanglement_bond(state) for state in qk_results_list])
    print(f" >> Ended in {time.time()-now}")
    ent_means = np.mean(aer_ent, axis=0)
    ent_std = np.std(aer_ent, axis=0)

    if get_statevector == False:
        qk_results_list = None

    return ent_means, ent_std, qk_results_list

def logger(data):
    """
    Printing data.

    Parameters
    ----------
    data : dict
        Simulation details

    Returns
    -------
    None
    """

    print("===============================")
    print("SIMULATION DETAILS")
    for k,v in zip(data.keys(), data.values()):
        print(f"{k} = {v}", end="\n")
    print("===============================")

def main(ansatz = None, backend = 'Aer', get_statevector = False):
    """
    Main method to perform the computation, given the simulation
    details of the ansatz

    Parameters
    ----------
    ansatz : :py:class:`QuantumCircuit`, optional
       Quantum circuit with additional metadata, by default None
    backend : str, optional
        backend of the simulation. Possible: 'MPS', 'Aer', by default 'Aer'
    get_statevector : bool, optional
        If True, returns the statevector, by default False

    Returns
    -------
    array-like of floats
        average of the entanglement entropy of the ansatz along all bipartitions
    array-like of floats
        standard deviation of the entanglement entropy of the ansatz along all bipartitions
    array-like of floats
        average of the entanglement entropy of a Haar circuit along all bipartitions
    array-like of floats or None
        statevector of the system
    """

    metadata = ansatz.metadata
    logger(metadata)

    num_qubits = metadata['num_qubits']

    try:
        max_bond_dim = metadata['max_bond_dim']
    except: 
        max_bond_dim = 1_024
             

    ######################################################
    # GENERATE RANDOM PARAMETERS (both inputs and weights)
    trials = 1_00
    random_params = np.pi * np.random.rand(trials, len(ansatz.parameters))

    ######################################################
    # SIMULATION WITH MPS or Aer
    if backend == 'MPS':    
        ent_means, ent_std = mps_simulation(ansatz, random_params, max_bond_dim)

    elif backend == 'Aer':  
            ent_means, ent_std, statevectors = aer_simulation(ansatz, random_params, get_statevector=get_statevector)

    else:
        raise TypeError(f"Backend {backend} not available")

    ######################################################
    # ENTANGLEMENT STUDY
    print("Measured entanglement =     ", np.round(ent_means, 4))

    # Expected entanglement accross cut lines if Haar distributed
    ent_haar = haar_bond_entanglement(num_qubits)
    print("Haar entanglement at bond = ", np.round(ent_haar, 4))

    ######################################################

    return ent_means, ent_std, ent_haar, statevectors


if __name__ == '__main__': 

    # Fixing seed for reproducibility
    seed = 34
    np.random.seed(seed)

    # Quantum Cirucit structure
    num_qubits = 4
    num_reps = 4
    alternate = True

    # Select a feature map and a variational block
    # Reps: Be sure that they use just one rep, as they are used as building blocks to build the full circuit. 
    # Inputs: Parameters in the feature_map are considered like inputs, so are equal in each layer.
    # Weights: Parameters in the var_ansata are considered trainable variables, so are different in each layer.
    #ansatz = ring_circ(num_qubits, num_reps=num_reps, barrier=False)  # Ring circ
    #ansatz = Abbas_QNN(num_qubits, reps = num_reps, alternate = alternate, barrier = True) # AbbassQNN
    feature_map = ZZFeatureMap(num_qubits, reps=1, entanglement='linear')
    var_ansatz = TwoLocal(num_qubits, 'ry', 'cx', 'linear', reps=1, insert_barriers=True, skip_final_rotation_layer=True)
    ansatz = general_qnn(num_reps, feature_map = feature_map, var_ansatz = var_ansatz, alternate = alternate, barrier = False)

    # Choose simulation backend
    #backend = 'MPS'
    backend = 'Aer'

    main(ansatz, backend=backend)
