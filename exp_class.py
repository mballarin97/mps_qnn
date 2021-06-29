from qiskit import QuantumCircuit
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.algorithms.optimizers import COBYLA, ADAM
from qiskit.circuit.library import TwoLocal, ZZFeatureMap
from qiskit_machine_learning.algorithms import VQC
from qiskit.providers.aer import QasmSimulator

import numpy as np

class QNN_experiment():
    
    def __init__(self, num_qub, freps=1, areps=1, seed=123, bond_dim=2, epochs=[50, 50]):
        """
            Parameters
            ----------
            num_qub : int
                Number of qubits
            freps : int
                Repetitions of the feature map
            areps : int
                Repetition of the ansatz
            seed : int
                Random seed for the experiment
            bond_dim : int
                Maximum bond dimension
            epochs : list of int, shape (1, 2)
                Number of max epochs for the mps and the statevector respectively. If 
                any of those number is 0 then that part is not computed
        """
        
        self.num_qub = num_qub
        self.seed = seed
        self.epochs = epochs
        self.areps = areps
        self.freps = freps
        self.bond_dim = bond_dim
        
        self.feature_map = ZZFeatureMap(feature_dimension=num_qub, 
                                        reps=freps, entanglement='linear', insert_barriers=True)
        self.ansatz = TwoLocal(num_qub, ['ry'], 'cx', reps=areps, entanglement='linear')
        
        self.statevect_sim = QasmSimulator(max_parallel_experiments=0, statevector_parallel_threshold=2, 
                                           method="statevector")
        
        self.mps_sim = QasmSimulator(method='matrix_product_state', 
                                     matrix_product_state_max_bond_dimension=bond_dim)
        
    def _init_statevector(self, init_params=[]):
        """
            Initialize the statevector-simulated VQC
        """
        self.vqc_sv = VQC(feature_map      = self.feature_map,
                          ansatz           = self.ansatz,
                          optimizer        = COBYLA(maxiter=self.epochs[1] ),
                          quantum_instance = QuantumInstance(self.statevect_sim,
                                                             shots=1024,
                                                             seed_simulator=self.seed,
                                                             seed_transpiler=self.seed),
                          warm_start       = True
                         )
        if len(init_params) != 0:
            self.vqc_sv._fit_result = [init_params ]
        
    def _init_mps(self, init_params=[]):
        """
            Initialize the mps-simulated VQC
        """
        self.vqc_mps = VQC(feature_map      = self.feature_map,
                           ansatz           = self.ansatz,
                           optimizer        = COBYLA(maxiter=self.epochs[0] ),
                           quantum_instance = QuantumInstance(self.mps_sim,
                                                             shots=1024,
                                                             seed_simulator=self.seed,
                                                             seed_transpiler=self.seed),
                           warm_start       = True
                         )
        if len(init_params) != 0:
            self.vqc_mps._fit_result = [init_params ]
        
    def _set_seeds(self):
        """
            Set random seeds
        """
        algorithm_globals.random_seed = self.seed
        np.random.seed(self.seed)
        
    def fit(self, features, labels, sim='statevector'):
        """
            Fit the data using the selected simulator
            
            Parameters
            ----------
            features: np.array
                Data features
            labels: np.array
                Data labels
            sim: string
                If statevector use the statevector simulator,
                If mps the mps simulator
        """
        
        if sim == 'statevector':
            self.vqc_sv.fit(features, labels)
        elif sim == 'mps':
            self.vqc_mps.fit(features, labels)
            
    def score(self, features, labels, sim='statevector'):
        """
            Obtain the accuracy the data using the selected simulator
            
            Parameters
            ----------
            features: np.array
                Data features
            labels: np.array
                Data labels
            sim: string
                If statevector use the statevector simulator,
                If mps the mps simulator
        """
        
        if sim == 'statevector':
            accuracy = self.vqc_sv.score(features, labels)
        elif sim == 'mps':
            accuracy = self.vqc_mps.score(features, labels)
            
        return accuracy
    
    def predict(self, features, sim='statevector'):
        """
            Predict the labels of the input features, according to a simulator
            
            Parameters
            ----------
            features: np.array
                Data features
            sim: string
                If statevector use the statevector simulator,
                If mps the mps simulator
            
            Returns
            -------
            labels : np.array
                Predicted labels
        """
        if sim == 'statevector':
            labels = self.vqc_sv.predict(features)
        elif sim == 'mps':
            labels = self.vqc_mps.predict(features)
            
        return accuracy
    
        
    
    def experiment(self, features, labels, init_params=[]):
        """
            Run the experiment with the first epochs[0] on the mps as pretrain and 
            the second epochs[1] on the statevector.
            
            Parameters
            ----------
            features: np.array
                Data features
            labels: np.array
                Data labels
            init_params: np.array
                Initial parameters. If none, they are chosen uniformly in [-1,1]
                
            Return
            ------
            scores: list
                score of the MPS and the statevector QNN
            
        """
        self._set_seeds()
        if len(init_params)==0:
            init_params = np.random.uniform(-1, 1,  len(self.ansatz.parameters))
        
        
        if self.epochs[0]>0:
            self._init_mps(init_params=init_params)
            self.fit(features, labels, sim = 'mps')

            final_params = self.vqc_mps._fit_result[0]
            mps_score = self.score(features, labels, sim = 'mps')
        else:
            final_params = init_params
            mps_score = 0
        
        if self.epochs[1]>0:
            self._init_statevector(init_params=final_params)
            self.fit(features, labels, sim = 'statevector')
            sv_score = self.score(features, labels, sim = 'statevector')
        else:
            sv_score = 0
            
        scores = [mps_score, sv_score]
        return scores
        