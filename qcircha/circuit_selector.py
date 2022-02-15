"""
Multiple analysis of the entanglement in the MPS circuit
(alternate vs. non alternate, varying number of reps, entanglement saturation to haar states) and plots.
"""

# Import necessary modules
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, TwoLocal

from qcircha.circuits import *

"""
TODO: This function is not really clear. What does it do? Why?

def is_close(measured, theory):
    dist = np.linalg.norm(theory-measured)
    dist1 = np.mean(np.abs(ent_haar-ent_meas)) 
    # Sum of entanglement
    dist2 = np.sum(theory) - np.sum(measured)
    return dist2, dist1
"""


def pick_circuit(num_qubits, num_reps, feature_map = 'ZZFeatureMap',
                 var_ansatz = 'TwoLocal', alternate=True):
    """
    Select a circuit with a feature map and a variational block. Examples below.
    Each block must have reps = 1, and then specify the correct number of repetitions
    is ensured by the :py:func:`general_qnn`.

    Available circuits:

    - 'ZZFeatureMap' : circuit with linear entanglement, used as feature map in the Power of Quantum Neural networks by Abbas et al.
    - 'TwoLocal' : circuit with linear entanglement, used as ansatz in the Power of Quantum Neural networks by Abbas et al.
    - 'Circuit15' : circuit with ring entanglement, defined in   (n.15 from Kim et al.)
    - 'Circuit12' : circuit with linear entanglement and a piramidal structure, defined in (n.12 from Kim et al.)
    - 'Circuit1' : circuit without entanglement with two single qubits rotations per (n.1 from Kim et al.)
    - 'Circuit9' : circuit number 9 from Kim et al.

    Parameters
    ----------
    num_qubits : int
        Number of qubits
    num_reps : int
        Number of repetitions
    feature_map : str or :py:class:`QuantumCircuit`, optional
        Type of feature map. Available options in the description. If a
        :py:class:`QuantumCircuit` it is used instead of the default ones.
        Default to 'ZZFeatureMap'.
    ansatz : str or :py:class:`QuantumCircuit`, optional
        Type of feature map. Available options in the description. If a
        :py:class:`QuantumCircuit` it is used instead of the default ones.
        Default to 'TwoLocal'.
    alternate : bool, optional
        If the feature map and the variational ansatz should be alternated in the
        disposition (True) or if first apply ALL the feature map repetitions and
        then all the ansatz repetitions (False). Default to True.

    Return
    ------
    :py:class:`QuantumCircuit`
        quantum circuit with the correct structure
    """

    feature_map = _select_circ(num_qubits, feature_map)
    var_ansatz = _select_circ(num_qubits, var_ansatz)

    # Build the PQC
    ansatz = general_qnn(num_reps, feature_map=feature_map,
                         var_ansatz=var_ansatz, alternate=alternate, barrier=False)

    return ansatz


def _select_circ(num_qubits, circ = 'ZZFeatureMap'):
    """
    Select the circuit based on the possibilities

    Available circuits:
    - 'ZZFeatureMap' : circuit with linear entanglement, used as feature map
        in the Power of Quantum Neural networks by Abbas et al.
    - 'TwoLocal' : circuit with linear entanglement, used as ansatz in
        the Power of Quantum Neural networks by Abbas et al.
    - 'Circuit15' : circuit with ring entanglement, defined in  
        (n.15 from Kim et al.)
    - 'Circuit12' : circuit with linear entanglement and a piramidal
        structure, defined in (n.12 from Kim et al.)
    - 'Circuit1' : easy circuit without entanglement (n.1 from Kim et al.) 
    - 'circuit9' : circuit n. 9 from Kim et al.

    Parameters
    ----------
    num_qubits : int
        Number of qubits
    circ : str or :py:class:`QuantumCircuit`, optional
        Type of circuit. Available options in the description. If a
        :py:class:`QuantumCircuit` it is used instead of the default ones.
        Default to 'ZZFeatureMap'.
    
    Return
    ------
    :py:class:`QuantumCircuit`
        Selected quantum circuit
    """
    
    # If it is a quantum circuit, directly return that.
    # Otherwise, go through the list
    if not isinstance(circ, QuantumCircuit):
        circ = circ.lower()

        if circ == 'zzfeaturemap':
            circ = ZZFeatureMap(num_qubits, reps=1, entanglement='linear')

        elif circ == 'twolocal':
            circ = TwoLocal(num_qubits, 'ry', 'cx', 'linear', reps=1, insert_barriers=False, skip_final_rotation_layer=True)

        elif circ == 'circuit15':
            circ = circuit15(num_qubits, num_reps=1, barrier=False)

        elif circ == 'circuit12':
            circ = circuit12(num_qubits, num_reps=1, piramidal=True, barrier=False)
        
        elif circ == 'circuit9':
            circ = circuit9(num_qubits, num_reps=1, barrier=True)

        elif circ == 'circuit1':
            circ = circuit1(num_qubits, num_reps = 1, barrier = True)
        
        elif circ == 'identity':
            circ = identity(num_qubits)
        
        else:
            raise ValueError(f'Circuit {circ} is not implemented.')

    return circ


