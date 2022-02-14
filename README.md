# Quantum Circuit characterization

Repository with the code for the collaboration between Marco Ballarin, Riccardo Mengoni, Stefano Mangini.

Quantum circuit characterization (`qcircha`) contains the code necessary to characterize a variational quantum circuit
on the following aspects:

- the entanglement scaling
- the KL-divergence of the probability distribution of the outputs compared to the Haar distribution

Both the aspects are computed for an exact simulation with a reduced number of qubits using qiskit, and for a larger
number of qubits using an MPS simulator. This library so enable the user to characterize a variational quantum circuit
of almost any size compatible to the NISQ era.

## Requirement

To properly run these code you need the following packages:

- numpy
- qiskit
- tn_py_frontend
- qcomps

The latter two packages are available from Marco Ballarin upon reasonable request

## License

Decide a license