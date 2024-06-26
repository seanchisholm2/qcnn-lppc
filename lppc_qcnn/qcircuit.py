import copy
import sys
from scipy.linalg import expm

# PennyLane:
import pennylane as qml
from pennylane import numpy as np

from pennylane.templates import RandomLayers
# from qiskit import quantum_info as qi

# Plotting:
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
mpl.rcParams.update(mpl.rcParamsDefault)
from tqdm import tqdm
import csv

# Data/Modeling:
import math
import random
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from sklearn.preprocessing import OneHotEncoder

# TorchVision:
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


################### HELPER FUNCTIONS CLASS: Qubit Reduction ######################


# CLASS IMPORTS ("gellmann_ops.py"):
from gellmann_ops import GellMannOps as gell


class QubitHelper:
    """
    Helper functions for Qubit Reduction.
    
    Example Class Instantiation:
    -> layers_lppc_instance = LayersLPPC()
    -> helper = QubitHelper(layers_lppc_instance)
    -> loops_required = helper.required_loops()
    -> binary_labels = QubitHelper.bin_class(predicted_labels)
    """
    
    def __init__(self, layers_lppc):
        self.layers_lppc = layers_lppc

    
    # Required Pooling Loops:
    def required_loops(self):
        """
        Determines the number of loops required to reduce the number of active qubits to 1.
        """
        if self.layers_lppc.num_qubits < 1:
            raise ValueError("Number of active qubits must be at least 1.")
        return math.ceil(math.log2(self.layers_lppc.num_qubits))

    
    # Binary Labeling:
    @staticmethod
    def bin_class(y, threshold=0.5):
        """
        Converts predicted probabilities to binary classification (0 or 1).
        """
        bin_y = np.where(y >= threshold, 1, 0)
        return bin_y


################### QUANTUM CIRCUIT AND LAYERS CLASS ######################


class QCircuitLPPC:
    """
    Contains functions for different layers in a quantum convolutional neural network and defines quantum circuits.
    """
    def __init__(self):
        # Initialize GellMannOps to access its variables
        self.gell_ops = gell()
        self.n_qubits = self.gell_ops.n_qubits
        self.num_active_qubits = self.gell_ops.num_active_qubits
        self.num_qubits = self.gell_ops.num_qubits
        self.active_qubits = self.gell_ops.active_qubits

    # Convolution Layer (Version #1):
    def conv_layer_V1(self, weights, active_qubits=None):
        """
        Applies a quantum convolutional layer.
        """
        # Check 'active_qubits' is passed:
        if active_qubits is None:
            active_qubits = self.active_qubits

        # Ensure active_qubits is *list* of qubit indices:
        if isinstance(active_qubits, int):
            active_qubits = list(range(active_qubits))

        if np.ndim(weights) == 0:
            weights = np.array([weights])

        for i in range(0, active_qubits, 2):
            # First rotation on second qubit of pair:
            qml.RZ(-np.pi / 2, wires=i + 1)

            # First set of rotations on first qubit of pair:
            qml.RZ(weights[0], wires=i)
            qml.RY(weights[0], wires=i + 1)

            # First CNOT gate:
            qml.CNOT(wires=[i, i + 1])

            # Second set of rotations on second qubit of pair:
            qml.RY(weights[0], wires=i + 1)

            # Second CNOT gate:
            qml.CNOT(wires=[i + 1, i])

            # Second rotation on first qubit of pair:
            qml.RZ(np.pi / 2, wires=i)

    # Pooling Layer (Version #1):
    def pool_layer_V1(self, weights, active_qubits=None):
        """
        Applies two-qubit pooling operation to inputted qubits, including controlled rotation based on
        measurement (first version).
        """
        # Check 'active_qubits' is passed:
        if active_qubits is None:
            active_qubits = self.active_qubits

        # Generate Gell-Mann matrices for 2-D space (single qubit operators):
        pool_operators = self.gell_ops.generate_gell_mann(2)

        # Ensure active_qubits is *list* of qubit indices:
        if isinstance(active_qubits, int):
            active_qubits = list(range(active_qubits))

        # Loop over all active qubits (in pairs):
        for i in range(0, self.n_qubits, 2):
            q1 = active_qubits[i]   # First qubit
            q2 = active_qubits[i+1] # Second qubit

            # Get convolutional operators V1 and V2 from pool operators and weights:
            v1 = self.gell_ops.get_conv_op(pool_operators, weights[i])  # V1 -> First set of weights
            v2 = self.gell_ops.get_conv_op(pool_operators, weights[i+1])  # V2 -> Second set of weights

            # Apply Hadamard gate to first qubit:
            qml.Hadamard(wires=q1)

            # Apply first controlled unitary operation:
            qml.ControlledQubitUnitary(v1, control_wires=[q1], wires=[q2])

            # Apply Hadamard gate to second qubit:
            qml.Hadamard(wires=q2)

            # Apply second controlled unitary operation:
            qml.ControlledQubitUnitary(v2, control_wires=[q2], wires=[q1])

            # Perform PauliZ expectation value measurement on second qubit:
            # qml.expval(qml.PauliZ(q2))
            qml.measure(q2)

        # Update active qubits by pooling 1 out of every 2 qubits:
        self.active_qubits = active_qubits[::2]

    # Pooling Layer (Version #2):
    def pool_layer_V2(self, weights, active_qubits=None):
        """
        Applies two-qubit pooling operation to inputted qubits, including controlled rotation based on
        measurement (second version).
        """
        # Check 'active_qubits' is passed:
        if active_qubits is None:
            active_qubits = self.active_qubits

        # Generate Gell-Mann matrices for 2-D space (single qubit operators):
        pool_operators = self.gell_ops.generate_gell_mann(2)

        # Ensure active_qubits is *list* of qubit indices:
        if isinstance(active_qubits, int):
            active_qubits = list(range(active_qubits))

        # Loop over all active qubits (in pairs):
        index = 0
        while index + 2 < len(active_qubits):
            q1 = active_qubits[index]   # First qubit
            q2 = active_qubits[index+1] # Second qubit

            # Get convolutional operators V1 and V2 from pool operators and weights:
            v1 = self.gell_ops.get_conv_op(pool_operators, weights[index])  # V1 -> First set of weights
            v2 = self.gell_ops.get_conv_op(pool_operators, weights[index+1])  # V2 -> Second set of weights

            # Apply Hadamard Gate to first qubit:
            qml.Hadamard(wires=q1)

            # Apply first Controlled Unitary Operation:
            qml.ControlledQubitUnitary(v1, control_wires=[q1], target_wires=[q2])

            # Perform PauliZ expectation value measurement on second qubit:
            # qml.measure(q2)

            index += 2

        # Update active qubits by pooling 1 out of every 2 qubits:
        self.active_qubits = active_qubits[::2]

    # Pooling Layer (Version #3):
    def pool_layer_V3(self, weights, active_qubits=None):
        """
        Applies two-qubit pooling operation to inputted qubits, including controlled rotation based on
        measurement (second version).
        """
        # Check 'active_qubits' is passed:
        if active_qubits is None:
            active_qubits = self.active_qubits

        # Generate Gell-Mann matrices for 2-D space (single qubit operators):
        pool_operators = self.gell_ops.generate_gell_mann(2)

        # Ensure active_qubits is *list* of qubit indices:
        if isinstance(active_qubits, int):
            active_qubits = list(range(active_qubits))

        # Loop over all active qubits (in pairs):
        index = 0
        while index + 2 < len(active_qubits):
            q1 = active_qubits[index]   # First qubit
            q2 = active_qubits[index+1] # Second qubit

            # Get convolutional operators V1 and V2 from pool operators and weights:
            v1 = self.gell_ops.get_conv_op(pool_operators, weights[index])  # V1 -> First set of weights
            v2 = self.gell_ops.get_conv_op(pool_operators, weights[index+1])  # V2 -> Second set of weights

            # Apply Hadamard Gate to first qubit:
            qml.Hadamard(wires=q1)

            # Apply first Controlled Unitary Operation:
            qml.ControlledQubitUnitary(v1, control_wires=[q1], target_wires=[q2])

            # Perform PauliZ expectation value measurement on second qubit:
            # qml.measure(q2)

            index += 2

        # Update active qubits by pooling 1 out of every 2 qubits:
        self.active_qubits = active_qubits[::2]

    # Fully Connected Layer (Version #1):
    def fully_connected_layer_V1(self, params, active_qubits=None):
        """
        Applies a fully connected layer to the remaining active qubits (first version).
        """
        # Check 'active_qubits' is passed:
        if active_qubits is None:
            active_qubits = self.active_qubits

        # Ensure active_qubits is a list of qubit indices:
        if isinstance(active_qubits, int):
            active_qubits = list(range(active_qubits))

        num_qubits = len(active_qubits)  # Initialize active qubits

        # Apply Fully Connected Operator to active qubits:
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                qml.CNOT(wires=[active_qubits[i], active_qubits[j]])
                qml.RY(params[i][j], wires=active_qubits[j])
                qml.CNOT(wires=[active_qubits[i], active_qubits[j]])

    # Fully Connected Layer (Version #2):
    def fully_connected_layer_V2(self, params, active_qubits=None):
        """
        Applies a fully connected layer to the remaining active qubits (second version).
        """
        # Check 'active_qubits' is passed:
        if active_qubits is None:
            active_qubits = self.active_qubits

        # Ensure active_qubits is *list* of qubit indices:
        if isinstance(active_qubits, int):
            active_qubits = list(range(active_qubits))

        # Generate Gell-Mann matrices for vector space:
        fc_mats = self.gell_ops.generate_gell_mann(2**len(active_qubits))
        fc_op = self.gell_ops.get_conv_op(fc_mats, params)

        # Apply Fully Connected Operator to active qubits:
        qml.QubitUnitary(fc_op, wires=active_qubits)

    # Quantum Circuit (Version #1):
    def q_circuit_V1(self, params, x, active_qubits=None):
        """
        Defines a quantum circuit for a Quantum Convolutional Neural Network (QCNN). Encodes
        input features using amplitude embedding, applies a convolutional layer, a pooling layer, and a fully
        connected layer, and then measures the qubits (first version).
        """
        # Check 'active_qubits' is passed:
        if active_qubits is None:
            active_qubits = self.active_qubits

        # Apply Amplitude Embedding:
        qml.AmplitudeEmbedding(x, wires=range(self.n_qubits), normalize=True)

        # Apply first convolutional layer function, pass 'params' as argument:
        self.conv_layer_V1(params, active_qubits)

        # Apply pooling layer, pass 'params' as argument:
        self.pool_layer_V3(params, active_qubits)

        # Apply fully connected layer, pass 'params' as argument:
        self.fully_connected_layer_V2(params, active_qubits)

        # Measure remaining active qubits:
        return [qml.measure(wire) for wire in active_qubits]

    # Quantum Circuit (Version #2):
    def q_circuit_V2(self, params, x, active_qubits=None):
        """
        Defines a quantum circuit for a Quantum Convolutional Neural Network (QCNN). Encodes
        input features using amplitude embedding, applies a convolutional layer, a pooling layer, and a fully
        connected layer, and then measures the qubits.
        """
        # Check 'active_qubits' is passed:
        if active_qubits is None:
            active_qubits = self.active_qubits

        # Apply Amplitude Embedding:
        qml.AmplitudeEmbedding(x, wires=range(self.n_qubits), normalize=True)

        # Apply first convolutional layer function, pass 'params' as argument:
        self.conv_layer_V1(params, active_qubits)

        # Apply pooling layer, pass 'params' as argument:
        self.pool_layer_V3(params, active_qubits)

        # Apply fully connected layer, pass 'params' as argument:
        self.fully_connected_layer_V2(params, active_qubits)

        # Measure remaining active qubits:
        return qml.expval(qml.PauliZ(active_qubits))


################### OPTIMIZATION AND COST CLASS ######################


class OptStepLPPC(QCircuitLPPC):
    """
    Contains functions for optimization steps in a Quantum Convolutional Neural Network (QCNN).
    It depends on the layer functions provided in the QCircuitLPPC class.
    
    Example Class Instantiation (And Dependencies): 
    -> opt_step = OptStepLPPC()
    -> updated_params, avg_cost = opt_step.stoch_grad_descent(params, x, y)
    """

    def __init__(self, bin_class):
        super().__init__()
        self.learning_rate = 0.01
        self.num_steps = 100
        self.batch_size = 10
        self.gen_vars_toCheck = ['x_train', 'x_test', 'y_train', 'y_test', 'qcnn_weights']  # General
        self.batch_vars_toCheck = ['x_train', 'x_test', 'y_train', 'y_test', 'qcnn_weights',
                                   'batch_cost', 'total_cost']  # Batch
        self.loss_vars_toCheck = ['x_train', 'x_test', 'y_train', 'y_test', 'qcnn_weights',
                                  'loss_history']  # Loss
        self.step_size = 5
        self.loss_history = []
        self.active_qubits = self.num_active_qubits
        self.n_qubits = self.n_qubits_mnist
        self.bin_class = bin_class

    # Mean Squared Error:
    def mse_cost(self, params, x, y, active_qubits=None, n_qubits=None):
        """
        Computes the Mean Squared Error (MSE) cost function.
        """
        # Check 'active_qubits' is passed:
        if active_qubits is None:
            active_qubits = self.active_qubits
        
        # Check 'n_qubits' is passed:
        if n_qubits is None:
            n_qubits = self.n_qubits

        predictions = np.array([self.q_circuit_V2(params, xi, active_qubits, n_qubits) for xi in x])
        return np.mean((self.bin_class(predictions) - y) ** 2)

    # Stochastic Gradient Descent:
    def stoch_grad_descent_V1(self, params, x, y, learning_rate, batch_size,
                              active_qubits=None, n_qubits=None):
        """
        Updates parameters using stochastic gradient descent and returns the updated 
        parameters and average cost.
        """
        # Check 'active_qubits' is passed:
        if active_qubits is None:
            active_qubits = self.active_qubits
        
        # Check 'n_qubits' is passed:
        if n_qubits is None:
            n_qubits = self.n_qubits

        # Shuffle Data:
        permutation = np.random.permutation(len(x))
        x = x[permutation]
        y = y[permutation]

        total_cost = 0  # Initialize Total Cost

        # Process each Batch:
        for i in range(0, len(x), batch_size):
            x_batch = x[i:i + batch_size]
            y_batch = y[i:i + batch_size]

            # Compute Gradient:
            grad_cost = qml.grad(self.mse_cost, argnum=0)
            gradients = grad_cost(params, x_batch, y_batch, active_qubits)  # RAM CRASH LOCATION

            # Update Parameters:
            params = params - learning_rate * gradients

            # Compute Cost for Batch:
            batch_cost = self.mse_cost(params, x_batch, y_batch, active_qubits)
            total_cost += batch_cost * len(x_batch)  # Accumulate total cost

        # Average Total Cost over all samples
        avg_cost = total_cost / len(x)

        return params, avg_cost


################### MAIN ######################

def main():
    return


if __name__ == "__main__":
    main()
