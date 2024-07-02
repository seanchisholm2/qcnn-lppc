########################################## qcircuit.py ############################################

### IMPORTS / DEPENDENCIES:

# PennyLane:
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers

# Data/Modeling:
import math
import random
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder

# Other:
from scipy.linalg import expm


################### QUANTUM CIRCUIT AND LAYERS CLASS ######################


# CLASS IMPORTS ("gellmann_ops.py"):
from gellmann_ops import GellMannOps as gell_ops

class QCircuitLPPC:
    """
    Contains functions for different layers in a quantum convolutional neural network and defines quantum circuits.
    """
    def __init__(self):
        # Initialize GellMannOps to access variables:
        self.gell_ops = GellMannOps()
        self.n_qubits = self.gell_ops.n_qubits
        self.num_active_qubits = self.gell_ops.num_active_qubits
        self.num_qubits = self.gell_ops.num_qubits
        self.active_qubits = self.gell_ops.active_qubits

    # CONVOLUTIONAL LAYER (VERSION #1):
    def conv_layer_V1(self, weights, active_qubits=None):
        """
        Applies a quantum convolutional layer (FIRST version).
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
            
    # CONVOLUTIONAL LAYER (VERSION #2):
    def conv_layer_V2(self, weights, active_qubits=None):
        """
        Applies a quantum convolutional layer (SECOND version).
        """
        #------------------------------------------------------
        # Check 'active_qubits' is passed:
        if active_qubits is None:
            active_qubits = self.active_qubits
        
        # Ensure active_qubits is *list* of qubit indices:
        if isinstance(active_qubits, int):
            active_qubits = list(range(active_qubits))
        #------------------------------------------------------

        if np.ndim(weights) == 0:
            weights = np.array([weights])

        # Generate Gell-Mann matrices for 2-D space (single qubit operators):
        pool_operators = self.gell_ops.generate_gell_mann(2)

        # Loop over all active qubits (in pairs):
        index = 0
        while index + 2 < len(active_qubits):
            q1 = active_qubits[index]   # First qubit
            q2 = active_qubits[index+1] # Second qubit

            # Get Convolutional Operator 'v_conv' from 'get_conv_op' and weights:
            v_conv = self.gell_ops.get_conv_op(pool_operators, weights[index])  # V -> Conv. Operator
            
            # Apply Controlled Unitary Operation with convolutional operator:
            qml.ControlledQubitUnitary(v_conv, control_wires=[q1], wires=[q2])

            index += 2

    # POOLING LAYER (VERSION #1):
    def pool_layer_V1(self, weights, active_qubits=None):
        """
        Applies two-qubit pooling operation to inputted qubits, including controlled rotation based on
        measurement (FIRST version).
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

    # POOLING LAYER (VERSION #2):
    def pool_layer_V2(self, weights, active_qubits=None):
        """
        Applies two-qubit pooling operation to inputted qubits, including controlled rotation based on
        measurement (SECOND version).
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

    # POOLING LAYER (VERSION #3):
    def pool_layer_V3(self, weights, active_qubits=None):
        """
        Applies two-qubit pooling operation to inputted qubits, including controlled rotation based on
        measurement (THIRD version).
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
        
    # POOLING LAYER (VERSION #4):
    def pool_layer_V4(self, weights, active_qubits=None):
        """
        Applies two-qubit pooling operation to inputted qubits, including controlled rotation based on
        measurement (FOURTH version).
        """
        #------------------------------------------------------
        # Check 'active_qubits' is passed:
        if active_qubits is None:
            active_qubits = self.active_qubits
        
        # Ensure active_qubits is *list* of qubit indices:
        if isinstance(active_qubits, int):
            active_qubits = list(range(active_qubits))
        #------------------------------------------------------

        # Generate Gell-Mann matrices for 2-D space (single qubit operators):
        pool_operators = self.gell_ops.generate_gell_mann(2)

        # Loop over all active qubits (in pairs):
        index = 0
        while index + 2 < len(active_qubits):
            q1 = active_qubits[index]   # First qubit
            q2 = active_qubits[index+1] # Second qubit

            # Get convolutional operators V1 and V2 from pool operators and weights:
            v1 = self.gell_ops.get_conv_op(pool_operators, weights[index])  # V1 -> First set of weights
            v2 = self.gell_ops.get_conv_op(pool_operators, weights[index+1])  # V2 -> Second set of weights
            
            # Get convolutional operators "V1_pool" and "v2_pool" from pool operators and weights:
            v1_pool = self.gell_ops.controlled_pool(v1)
            v2_pool = self.gell_ops.controlled_pool(v2)

            # Apply Hadamard Gate to first qubit:
            qml.Hadamard(wires=q1)

            # Apply FIRST Controlled Unitary Operation:
            qml.ControlledQubitUnitary(v1_pool, control_wires=[q1], wires=[q2])

            # Apply SECOND Controlled Unitary Operation:
            qml.ControlledQubitUnitary(v2_pool, control_wires=[q1], wires=[q2])

            index += 2

        # Update active qubits by pooling 1 / 2 qubits:
        self.active_qubits = active_qubits[::2]

    # FULLY CONNECTED LAYER (VERSION #1):
    def fully_connected_layer_V1(self, params, active_qubits=None):
        """
        Applies a fully connected layer to the remaining active qubits (FIRST version).
        """
        #------------------------------------------------------
        # Check 'active_qubits' is passed:
        if active_qubits is None:
            active_qubits = self.active_qubits
        
        # Ensure active_qubits is *list* of qubit indices:
        if isinstance(active_qubits, int):
            active_qubits = list(range(active_qubits))
        #------------------------------------------------------

        num_qubits = len(active_qubits)  # Initialize active qubits

        # Apply Fully Connected Operator to active qubits:
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                qml.CNOT(wires=[active_qubits[i], active_qubits[j]])
                qml.RY(params[i][j], wires=active_qubits[j])
                qml.CNOT(wires=[active_qubits[i], active_qubits[j]])

    # FULLY CONNECTED LAYER (VERSION #2):
    def fully_connected_layer_V2(self, params, active_qubits=None):
        """
        Applies a fully connected layer to the remaining active qubits (SECOND version).
        """
        #------------------------------------------------------
        # Check 'active_qubits' is passed:
        if active_qubits is None:
            active_qubits = self.active_qubits
        
        # Ensure active_qubits is *list* of qubit indices:
        if isinstance(active_qubits, int):
            active_qubits = list(range(active_qubits))
        #------------------------------------------------------

        # Generate Gell-Mann matrices for vector space:
        fc_mats = self.gell_ops.generate_gell_mann(2**len(active_qubits))
        fc_op = self.gell_ops.get_conv_op(fc_mats, params)

        # Apply Fully Connected Operator to active qubits:
        qml.QubitUnitary(fc_op, wires=active_qubits)

    # QUANTUM CIRCUIT (VERSION #1):
    def q_circuit_V1(self, params, x, active_qubits=None):
        """
        Defines a quantum circuit for a Quantum Convolutional Neural Network (QCNN). Encodes
        input features using amplitude embedding, applies a convolutional layer, a pooling layer, and a fully
        connected layer, and then measures the qubits (FIRST version).
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

    # QUANTUM CIRCUIT (VERSION #2):
    def q_circuit_V2(self, params, x, active_qubits=None):
        """
        Defines a quantum circuit for a Quantum Convolutional Neural Network (QCNN). Encodes
        input features using amplitude embedding, applies a convolutional layer, a pooling layer, and a fully
        connected layer, and then measures the qubits (SECOND version).
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
    
    # QUANTUM CIRCUIT (VERSION #3):
    def q_circuit_V3(self, params, x, active_qubits=None):
        """
        Defines a quantum circuit for a Quantum Convolutional Neural Network (QCNN). Encodes
        input features using amplitude embedding, applies a convolutional layer, a pooling layer, and a fully
        connected layer, and then measures the qubits (THIRD version).
        """
        #------------------------------------------
        # Check 'active_qubits' is passed:
        if active_qubits is None:
            active_qubits = self.active_qubits
        
        # Check 'n_qubits' is passed:
        if n_qubits is None:
            n_qubits = self.n_qubits
        #------------------------------------------

        # Apply Amplitude Embedding:
        qml.AmplitudeEmbedding(x, wires=range(n_qubits), normalize=True)
        
        # FIRST LAYER (1):
        #----------------------------------------------------------------
         # Apply Convolutional Layer (pass 'params' as argument):
        self.conv_layer_V2(params, active_qubits)

        # Apply Pooling Layer:
        self.pool_layer_V4(params, active_qubits)
        #----------------------------------------------------------------
       
        # SECOND LAYER (2):
        #----------------------------------------------------------------
         # Apply Convolutional Layer:
        self.conv_layer_V2(params, active_qubits)

        # Apply Pooling Layer:
        self.pool_layer_V4(params, active_qubits)
        #----------------------------------------------------------------
        
        # Apply Fully Connected Layer:
        self.fully_connected_layer_V2(params, active_qubits)

        # Measure middle qubit of remaining active qubits:
        middle_qubit = active_qubits[len(active_qubits) // 2]
        
        return qml.expval(qml.PauliZ(middle_qubit))


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
        self.max_iteration = 100
        self.conv_tol = 1e-06

        # MEMORY:
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

    # OPTIMIZER SELECTION:
    @staticmethod
    def qc_opt_select(opt_num):
        """
        Selects and returns the desired optimizer from the given list of optimizers, based on the given 
        value for 'opt_num'.
        Raises: "ValueError: If opt_num is not 1, 2, or 3".

        Example Function Usage:
        -> qc_opt_print()
        """
        # List of potential optimizers (3 total):
        if opt_list is None:
            opt_list = {
                1: "qml.GradientDescentOptimizer",
                2: "qml.AdamOptimizer",
                3: "qml.COBYLAOptimizer"
            }

        if opt_num not in opt_list:
            raise ValueError("opt_num must be equal to 1, 2, or 3 for default. Use 'qc_opt_print' to check full list.")

        opt = optimizers[opt_num]()

        return opt

    # LIST AVAILABLE OPTIMIZERS:
    @staticmethod
    def qc_opt_print(opt_list=None):
        """
        Prints the list of optimizer options and their associated 'opt_num' values. Allows list of usable optimizers to be
        appended if necessary, but defaults to a list of optimizers including the Stochastic Gradient Descent (SGD) 
        Optimizer, the ADAM Optimizer, and the COBYLA Optimizer.

        Example Function Usage:
        -> optimizer = qc_opt_select(1) (Sets 'optimizer' equal to Stochastic Gradient Descent (SGD) Optimizer)
        """
        # Check if new optimizer list was passed, assign default list if not:
        if
        # List of potential optimizers (3 total):
        opt_list = {
            1: "qml.GradientDescentOptimizer",
            2: "qml.AdamOptimizer",
            3: "qml.COBYLAOptimizer"
        }

        # Print optimizer options:
        print("Available Optimizer Options:")
        print("-------")
        for num, name in opt_list.items():
            print(f"{num}: {name}")
            print("-------")
    
    # MEAN SQUARED ERROR (MSE) COST:
    def mse_cost(self, params, x, y, active_qubits=None, n_qubits=None):
        """
        Computes the Mean Squared Error (MSE) cost function.
        """
        #---------------------------------------
        # Check 'active_qubits' is passed:
        if active_qubits is None:
            active_qubits = self.active_qubits
        
        # Check 'n_qubits' is passed:
        if n_qubits is None:
            n_qubits = self.n_qubits
        #---------------------------------------

        predictions = np.array([self.q_circuit_V2(params, xi, active_qubits, n_qubits) for xi in x])
        return np.mean((self.bin_class(predictions) - y) ** 2)

    # STOCHASTIC GRADIENT DESCENT (VERSION #1):
    def stoch_grad_descent_V1(self, params, x, y, learning_rate, batch_size,
                              active_qubits=None, n_qubits=None):
        """
        Updates parameters using stochastic gradient descent and returns the updated 
        parameters and average cost (FIRST version).
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

        # Initialize Total Cost:
        total_cost = 0

        # Process each Batch:
        for i in range(0, len(x), batch_size):
            x_batch = x[i:i + batch_size]
            y_batch = y[i:i + batch_size]

            # Compute Gradient:
            grad_cost = qml.grad(self.mse_cost, argnum=0)
            gradients = grad_cost(params, x_batch, y_batch, active_qubits)

            # Update Parameters:
            params = params - learning_rate * gradients

            # Compute Cost for Batch:
            batch_cost = self.mse_cost(params, x_batch, y_batch, active_qubits)
            total_cost += batch_cost * len(x_batch)  # Accumulate Total Cost

        # Average Total Cost over all samples:
        avg_cost = total_cost / len(x)

        return params, avg_cost
    
    # STOCHASTIC GRADIENT DESCENT OPTIMIZATION (VERSION #2):
    def stoch_grad_V2(self, opt, mse_cost, params, x, y, learning_rate, batch_size, max_iterations, conv_tol,
                     active_qubits=None, n_qubits=None):
        """
        Updates parameters using stochastic gradient descent and returns the updated parameters
        and average cost (SECOND version).
        """
        #---------------------------------------
        # Check 'active_qubits' is passed:
        if active_qubits is None:
            active_qubits = self.active_qubits
        
        # Check 'n_qubits' is passed:
        if n_qubits is None:
            n_qubits = self.n_qubits
        #---------------------------------------
        
        # Shuffle Data:
        permutation = np.random.permutation(len(x))
        x = x[permutation]
        y = y[permutation]

        # Initialize Total Cost:
        total_cost = 0

        # Process each Batch:
        for i in range(0, len(x), batch_size):
            x_batch = x[i:i + batch_size]
            y_batch = y[i:i + batch_size]

            for n in range(max_iterations):
                params, prev_cost = opt.step_and_cost(self.mse_cost, params, x_batch, y_batch)

                # Compute Cost for Current Params:
                batch_cost = mse_cost(params, x_batch, y_batch)
                total_cost += batch_cost * len(x_batch)  # Accumulate Total Cost

                # Convergence Check:
                conv = np.abs(batch_cost - prev_cost)
                if n % 10 == 0:
                    print(f"Step {n}: Cost function = {batch_cost:.8f}")

                if conv <= conv_tol:
                    break

        # Average Total Cost over all samples:
        avg_cost = total_cost / len(x)
        
        return params, avg_cost
    
    # OPTIMIZATION TRAINING LOOP:
    def qc_train(self, opt_num, weights, x_train, y_train, num_steps=None, learning_rate=None, 
               batch_size=None, max_iter=None, conv_tol=None, hist=False):
        """
        Trains the QCNN model using the specified optimizer, training parameters, and imported datasets, and
        returns the weights trained on the quantum circuit model.
        """
        # Initialize default training parameters values if not provided:
        if num_steps is None:
            num_steps = self.num_steps
        if learning_rate is None:
            learning_rate = self.learning_rate
        if batch_size is None:
            batch_size = self.batch_size
        if max_iter is None:
            max_iter = self.max_iteration
        if conv_tol is None:
            conv_tol = self.conv_tol

        # Select desired optimizer:
        opt = self.qc_opt_select(opt_num)

        loss_history = []

        # Training loop:
        for step in range(num_steps):
            weights, loss = self.stoch_grad_V2(opt, self.mse_cost, weights, x_train, y_train,
                                                    learning_rate, batch_size, max_iter, conv_tol)

            loss_history.append(loss)  # Accumulate loss

            # Print step and cost:
            print(f"Step {step}: cost = {loss}")

        loss_history = np.array(loss_history)   

        # Return trained parameters (and loss history if requested):
        if hist == False:
            return weights
        else:
            return weights, loss_history

    # ACCURACY (VERSION #1):
    @staticmethod
    def accuracy_V1(predictions, y):
        """
        Calculates the accuracy of the QCNN model on the provided testing data. Assumes predictions were calculated
        already, not dependent on quantum circuit function (FIRST version).
        """
        # Calculate Number of Correct Predictions:
        true_predictions = np.sum(predictions == y)

        # Calculate Accuracy:
        accuracy = true_predictions / len(y)

        return accuracy


################### MAIN ######################

def main():
    return


if __name__ == "__main__":
    main()
