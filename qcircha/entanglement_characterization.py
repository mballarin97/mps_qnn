"""
Contains main computational methods
"""

# Import necessary modules
import time

from qcomps import run_simulation
from qcomps.qk_utils import qk_transpilation_params
from qcomps import QCConvergenceParameters
import numpy as np

from qiskit import Aer, transpile

from qcircha.entanglement.haar_entanglement import haar_bond_entanglement
from qcircha.entanglement.statevector_entanglement import entanglement_bond
from qcircha.utils import logger

__all__ = ['entanglement_characterization']

def _run_mps(qc, max_bond_dim=1024, do_statevector=False):
    """
    Runs a quantum circuit (parameters already numerical) with MPS.

    Parameters
    ----------
    qc : :py:class:`QuantumCircuit`
        qiskit quantum circuit class
    max_bond_dim: int, optional
        Maximum bond dimension for the MPS. Default to 1024
    do_statevector : bool, optional
        If True, compute the statevector of the system. Default to None.

    Returns
    -------
    :py:class:`qcomps.simulation_results`
        The results of the simulations, comprehensive of the entanglement
    """

    conv_params = QCConvergenceParameters(max_bond_dim)
    trans_params = qk_transpilation_params(linearize=True)

    results = run_simulation(qc, convergence_parameters=conv_params,
                             transpilation_parameters=trans_params, do_entanglement=True,
                             approach='PY', save_mps='N', do_statevector=do_statevector)

    return results

def _run_circuit(qc, parameters, max_bond_dim=1024, do_statevector=False):
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
    do_statevector : bool, optional
        If True, compute the statevector of the system. Default to None.

    Returns
    -------
    :py:class:`qcomps.simulation_results`
        The results of the simulations, comprehensive of the entanglement
    """

    qc = qc.assign_parameters(parameters)
    results = _run_mps(qc, max_bond_dim=max_bond_dim, do_statevector=do_statevector)
    return results

def _mps_simulation(qc, random_params, max_bond_dim=1024, do_statevector=False):
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
    do_statevector : bool, optional
        If True, compute the statevector of the system. Default to None.

    Returns
    -------
    array-like of floats
        Average of the entanglement over different parameter sets
    array-like of floats
        Standard deviation of the entanglement over different parameter sets
    array-like of ndarray
        Array of statevectors if do_statevector=True, otherwise array of None
    """
    sim_bknd = Aer.get_backend('statevector_simulator')
        
    mps_results_list = []
    statevector_list = []
    for idx, params in enumerate(random_params):
        print(f"Run {idx}/{len(random_params)}", end="\r")
        qc = transpile(qc, sim_bknd) # Why this transpilation?
        mps_results = _run_circuit(qc, params, max_bond_dim=max_bond_dim,
            do_statevector=do_statevector)
        
        mps_results_list.append(mps_results)

    mps_entanglement = np.array([res.entanglement for res in mps_results_list])
    ent_means = np.mean(mps_entanglement, axis=0)
    ent_std = np.std(mps_entanglement, axis=0)
    statevector_list = [res.statevector for res in mps_results_list]

    return ent_means, ent_std, statevector_list

def _aer_simulation(qc, random_params, get_statevector = False):
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

def entanglement_characterization(ansatz = None, backend = 'Aer', get_statevector = False,
    **kwargs):
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
        if 'max_bond_dim' in kwargs:
            max_bond_dim = kwargs['max_bond_dim']
        ent_means, ent_std, statevectors = _mps_simulation(ansatz, random_params, max_bond_dim, do_statevector=get_statevector)

    elif backend == 'Aer':  
            ent_means, ent_std, statevectors = _aer_simulation(ansatz, random_params, get_statevector=get_statevector)

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
