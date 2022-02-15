"""
Compute the entanglement of a statevector
"""

import numpy as np

__all__ = ['get_reduced_density_matrix', 'entanglement_entropy', 'von_neumann_entropy',
    'entanglement_bond']

def get_reduced_density_matrix(psi, loc_dim, n_sites, sites, 
    print_rho=False):
    """
    Parameters
    ----------
    psi : ndarray
        state of the QMB system
    loc_dim : int
        local dimension of each single site of the QMB system
    n_sites : int
        total number of sites in the QMB system
    site : int or array-like of ints
        Indeces to trace away
    print_rho : bool, optional
        If True, it prints the obtained reduced density matrix], by default False

    Returns
    -------
    ndarray
        Reduced density matrix
    """
    if not isinstance(psi, np.ndarray):
        raise TypeError(f'density_mat should be an ndarray, not a {type(psi)}')

    if not np.isscalar(loc_dim) and not isinstance(loc_dim, int):
        raise TypeError(f'loc_dim must be an SCALAR & INTEGER, not a {type(loc_dim)}')

    if not np.isscalar(n_sites) and not isinstance(n_sites, int):
        raise TypeError(f'n_sites must be an SCALAR & INTEGER, not a {type(n_sites)}')

    if np.isscalar(sites):
        sites = [sites]
    
    # RESHAPE psi
    psi_copy=psi.reshape(*[loc_dim for _ in range(n_sites)])
    # DEFINE A LIST OF SITES WE WANT TO TRACE OUT
    indices=np.array(sites)

    # COMPUTE THE REDUCED DENSITY MATRIX
    rho=np.tensordot(psi_copy, np.conjugate(psi_copy), axes=(indices, indices))
    # PRINT RHO
    if print_rho:
        print('----------------------------------------------------')
        print(f'DENSITY MATRIX OF SITE ({str(sites)})')
        print('----------------------------------------------------')
        print(rho)

    return rho

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

    num_sites = int( np.log2(len(statevector)) )

    # Construct density matrix
    partial_rho = get_reduced_density_matrix(statevector, 2, num_sites, idx_to_trace)

    # get eigenvalues of partial_rho
    eigvs, _ = np.linalg.eigh(partial_rho)

    ent_entropy = von_neumann_entropy(eigvs)

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

def von_neumann_entropy(eigvs):
    """
    Compute the Von Neumann entanglement entropy of a density matrix
    with eigenvalues :math:`\\lambda_i`

    .. math::
        S_V = -Tr(\\rho\ln\\rho)=-\\sum_{i} \\lambda_i\\log \\lambda_i

    Parameters
    ----------
    eigvs : array-like of floats
        Eigenvalues of the density matrix
    
    Returns
    -------
    float
        entanglement entropy
    """

    eigvs = np.array(eigvs)
    entanglement = -np.sum( eigvs*np.log(eigvs) )

    return entanglement