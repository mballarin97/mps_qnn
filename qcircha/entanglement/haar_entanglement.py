"""
Compute the entanglement of a Haar circuit
"""

import numpy as np

__all__ = ['haar_bond_entanglement', 'haar_entanglement']

def _harmonic(n):
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
    ent = _harmonic(2**num_qubits) - _harmonic(2**(num_qubits - num_A)) - d
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

def haar_discrete(xmin, delta, n):
    """
    Generate a discrete distribution of Haar probabilities

    TODO : comment function

    Parameters
    ----------
    xmin : _type_
        Minimum x value
    delta : _type_
        _description_
    n : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    return (1. - xmin)**(n-1) - (1 - xmin - delta)**(n-1)