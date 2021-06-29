import numpy as np
from exp_class import QNN_experiment
from qiskit_machine_learning.datasets import iris

# ---------------------- DATA PREPARATION ----------------
# Use Iris data set for training and test data
feature_dim = 4  # dimension of each data point
training_size = 100
test_size = 30

# training features, training labels, test features, test labels as np.array,
# one hot encoding for labels
training_features, training_labels, test_features, test_labels = \
    iris(training_size=training_size, test_size=test_size, n=feature_dim)
# Eliminate third class
training_features = training_features[ training_labels[:, 2]!=1 ]
training_labels = training_labels[ training_labels[:, 2]!=1, :2 ]

test_features = test_features[ test_labels[:, 2]!=1 ]
test_labels = test_labels[ test_labels[:, 2]!=1, :2 ]
# ---------------------- END DATA PREPARATION ----------------

max_epochs = 20
chi = 1

stv_epochs = np.arange(0, max_epochs, 2)
mps_epochs = np.arange(0, max_epochs+1, 5)

# data structure mps_epoch/statevect_epoch/mps_score/statevect_score
data = np.zeros( (len(mps_epochs), max_epochs, 3) )

for i, epoch in enumerate(mps_epochs):
    print(f'---------- MPS epochs: {epoch} -----------')
    for j, stv_epoch in enumerate(range(max_epochs)):
        exp = QNN_experiment(feature_dim,  epochs=[epoch, stv_epoch], bond_dim=chi)
        exp_score = exp.experiment(training_features, training_labels)
        data[i, j, :] = [stv_epoch, *exp_score]
        print(f'--- stv epochs: {stv_epoch} ---')
        print(f'--- MPS score: {np.round(exp_score[0], 3)}, stv score: {np.round(exp_score[1], 3)} ---')

for j, ep in enumerate(mps_epochs):
    np.savetxt(f'data/iris-scores_mps-ep{ep}_bd{chi}.npy', data[j, :, :])