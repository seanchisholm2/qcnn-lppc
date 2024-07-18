########################################## CIRCUIT_LAYERS.PY ############################################

### IMPORTS / DEPENDENCIES:

# PennyLane:
import pennylane as qml
from pennylane import numpy as np

# TorchVision (FOR DATA):
import torch
from torchvision import datasets, transforms  # (NOT ACCESSED)
from torch.utils.data import DataLoader  # (NOT ACCESSED)

# TensorFlow (FOR DATA):
# import tensorflow as tf
# from tensorflow.keras.datasets import mnist

# Package:
# from .operators import QuantumMathOps as qmo


# ==============================================================================================
#                                      QCNN LAYERS CLASS
# ==============================================================================================


class LayersQC:
    """
    Contains all relevant circuit and layer functions for the quantum convolutional neural network.
    """
    def __init__(self):
        # QUBITS:
        self.n_qubits = 6 # Set 'n_qubits' equal to desired qubit configuration (for us, 6)
        self.n_qubits_draw = 2 # 2 qubit config for DRAWQC
        # ACTIVE QUBITS:
        self.active_qubits = 6 # Set 'active_qubits' equal to 'n_qubits''
        # WIRES:
        self.num_wires = 2
        self.wires = 6
    
    # -----------------------------------------------------------
    #        ORIGINAL CIRCUIT AND LAYER FUNCTIONS (LPPC)
    # -----------------------------------------------------------

    # ******* Original Convolutional Layer *******:
    def conv_layer_orig(self, weights, active_qubits, n_qubits=None,
                        check_qubits=True):
        """
        Applies a quantum convolutional layer (original version).
        """
        # Check Qubit Configuration:
        if check_qubits is True:
            # Number of Qubits:
            if n_qubits is None:
                n_qubits = self.n_qubits

        # Generate Gell-Mann matrices for 2-D space (single qubit operators):
        pool_operators = gell_ops.generate_gell_mann(gell_ops, (2**n_qubits))

        # Loop over all active qubits (in pairs):
        index = 0
        while index + 2 < len(active_qubits):
            q1 = active_qubits[index]   # First qubit
            q2 = active_qubits[index+1] # Second qubit

            # Get Convolutional Operator 'v_conv' from 'get_conv_op' and weights:
            v_conv = gell_ops.get_conv_op(gell_ops,
                                               pool_operators, weights[index])  # V -> Conv. Operator
            
            # Apply Controlled Unitary Operation with convolutional operator:
            qml.ControlledQubitUnitary(v_conv, control_wires=[q1], wires=[q2])

            index += 2

    # ******* Original Pooling Layer *******:
    def pool_layer_orig(self, weights, active_qubits, n_qubits=None,
                        check_qubits=True):
        """
        Applies two-qubit pooling operation to inputted qubits, including controlled rotation 
        based on a conditional measurement (original version).
        """
        # Check Qubit Configuration:
        if check_qubits is True:
            # Number of Qubits:
            if n_qubits is None:
                n_qubits = self.n_qubits
        
        # Generate Gell-Mann matrices for 2-D space (single qubit operators):
        pool_operators = gell_ops.generate_gell_mann(gell_ops, 2**len(n_qubits))

        # Loop over all active qubits (in pairs):
        index = 0
        while index + 2 < len(active_qubits):
            q1 = active_qubits[index]   # First qubit
            q2 = active_qubits[index+1] # Second qubit

            # Get convolutional operators V1 and V2 from pool operators and weights:
            # V1 -> First set of weights
            v1 = gell_ops.get_conv_op(gell_ops,
                                           pool_operators, weights[index])
            # V2 -> Second set of weights
            v2 = gell_ops.get_conv_op(gell_ops,
                                           pool_operators, weights[index+1])
            
            # Get convolutional operators "V1_pool" and "v2_pool" from pool operators and weights:
            v1_pool = gell_ops.controlled_pool(gell_ops, v1)
            v2_pool = gell_ops.controlled_pool(gell_ops, v2)

            # Apply Hadamard Gate to first qubit:
            qml.Hadamard(wires=q1)

            # Apply FIRST Controlled Unitary Operation:
            qml.ControlledQubitUnitary(v1_pool, control_wires=[q1], wires=[q2])

            # Apply SECOND Controlled Unitary Operation:
            qml.ControlledQubitUnitary(v2_pool, control_wires=[q1], wires=[q2])

            index += 2

        # Update active qubits by pooling 1 / 2 qubits:
        self.active_qubits = active_qubits[::2]


    # ******* Original Fully-Connected Layer *******:
    def fc_layer_orig(self, params, active_qubits, n_qubits=None,
                      check_qubits=True):
        """
        Applies a fully connected layer to the remaining active qubits (original version).
        """
        # Check Qubit Configuration:
        if check_qubits is True:
            # Number of Qubits:
            if n_qubits is None:
                n_qubits = self.n_qubits

        num_qubits = len(active_qubits)  # Initialize active qubits

        # Apply Fully Connected Operator to active qubits:
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                qml.CNOT(wires=[active_qubits[i], active_qubits[j]])
                qml.RY(params[i][j], wires=active_qubits[j])
                qml.CNOT(wires=[active_qubits[i], active_qubits[j]])


    # ******* Original Fully-Connected Layer (with Two-Qubit Unitaries) *******:
    def fc_layer_unitary(self, params, active_qubits, n_qubits=None,
                         check_qubits=True):
        """
        Applies a fully connected layer to the remaining active qubits, using two-qubit unitary
        operators (original version).
        """
        # Check Qubit Configuration:
        if check_qubits is True:
            # Number of Qubits:
            if n_qubits is None:
                n_qubits = self.n_qubits
        
        # Generate Gell-Mann matrices for vector space:
        fc_mats = gell_ops.generate_gell_mann(gell_ops, 2**len(n_qubits))
        fc_op = gell_ops.get_conv_op(gell_ops, fc_mats, params)

        # Apply Fully Connected Operator to active qubits:
        qml.QubitUnitary(fc_op, wires=active_qubits)

    # ******* Original Quantum Circuit *******:
    def qcircuit_orig(self, params, x, active_qubits=None, n_qubits=None,
                      check_qubits=True, draw=False):
        """
        Defines a quantum circuit for a Quantum Convolutional Neural Network (QCNN). Encodes
        input features using amplitude embedding, applies a convolutional layer, a pooling layer,
        and a fully connected layer, and then measures the qubits (original version).
        """
        # Check Qubit Configuration:
        if check_qubits is True:
            # Active Qubits:
            if active_qubits is None:
                active_qubits = self.active_qubits
                active_qubits = list(range(active_qubits))
            
            # Number of Qubits:
            if n_qubits is None:
                n_qubits = self.n_qubits

        # Amplitude Embedding:
        if draw is not True:
            qml.AmplitudeEmbedding(x, wires=range(n_qubits), normalize=True)
        
        # LAYER 1:
         # Convolutional Layer (pass 'params' as argument):
        self.conv_layer_V2(self, params, active_qubits)
        # Pooling Layer:
        self.pool_layer_V3(self, params, active_qubits)
       
        # LAYER 2:
         # Convolutional Layer:
        self.conv_layer_V2(self, params, active_qubits)
        # Pooling Layer:
        self.pool_layer_V3(self, params, active_qubits)
        
        # Fully Connected Layer (FINAL LAYER):
        self.fully_connected_layer_V2(self, params, active_qubits)

        # Measure *Middle Qubit* of remaining Active Qubits:
        middle_qubit = active_qubits[len(active_qubits) // 2]
        
        return qml.expval(qml.PauliZ(middle_qubit))
    
    # -----------------------------------------------------------
    #             NEW CIRCUIT AND LAYER FUNCTIONS
    # -----------------------------------------------------------

    def todo():
        """
        FILL.
        """
        # TO-DO

        return None


# =============================================================================================
#                                    CIRCUIT DRAWING CLASS
# =============================================================================================


class DrawQC(LayersQC):
    """
    Contains functions for drawing different layer functions relevant to the
    quantum convolutional neural network, including convolutional layers, pooling layers,
    fully connected layers, and quantum circuits. Defaults to a 2-Qubit representation of
    each layer, as well as defaulting to the most recent version of each layer function if
    none is specified.
    """
    def __init__(self):
        self.layers = LayersQC()
        # QUBITS:
        self.n_qubits = 6 # Set 'n_qubits' equal to desired qubit configuration (for us, 6)
        self.n_qubits_draw = 2 # 2 qubit config for DRAWQC
        # ACTIVE QUBITS:
        self.active_qubits = 6 # Set 'active_qubits' equal to 'n_qubits''
        # WIRES:
        self.num_wires = 2
        self.wires = 6

    # -----------------------------------------------------------
    #            ORIGINAL DRAWING FUNCTIONS (LPPC)
    # -----------------------------------------------------------

    # ******* Original Pooling Layer Drawing *******:
    def draw_pool_lppc(self, params, x, active_qubits=None, n_qubits=None,
                        check_qubits=True):
        """
        Draws the pooling layer functions for the LPPC QCNN package. Takes in a parameter 
        'version' to denote the version of the pooling layer that you want (3 total), and 
        defaults to the most recent version if nothing is passed. Uses qml.draw() to complete
        circuit diagrams.
        """
        # Check Qubit Configuration:
        if check_qubits is True:
            # Active Qubits:
            if active_qubits is None:
                active_qubits = self.active_qubits
                active_qubits = list(range(active_qubits))
            
            # Number of Qubits:
            if n_qubits is None:
                n_qubits = self.n_qubits

        # Initialize Device:
        dev = qml.device("default.qubit", wires=n_qubits)

        # CIRCUIT:
        @qml.qnode(dev)
        def pool_circuit(self, params, active_qubits):
            self.pool_layer_orig(self, params, active_qubits)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        # Draw and Print Circuit:
        drawing = qml.draw(pool_circuit, expansion_strategy="device")(self, params, active_qubits)
        print(drawing)


    # ******* Original Convolutional Layer Drawing *******:
    def draw_conv_lppc(self, params, x, active_qubits=None, n_qubits=None,
                        check_qubits=True):
        """
        Draws the convolutional layer functions for the LPPC QCNN package. Takes in a parameter 
        'version' to denote the version of the convolutional layer that you want (2 total), and 
        defaults to the most recent version if nothing is passed. Uses qml.draw() to complete
        circuit diagrams.
        """
        # Check Qubit Configuration:
        if check_qubits is True:
            # Active Qubits:
            if active_qubits is None:
                active_qubits = self.active_qubits
                active_qubits = list(range(active_qubits))
            
            # Number of Qubits:
            if n_qubits is None:
                n_qubits = self.n_qubits

        # Initialize Device:
        dev = qml.device("default.qubit", wires=n_qubits)

        # CIRCUIT:
        @qml.qnode(dev)
        def conv_circuit(self, params, active_qubits):
            self.conv_layer_orig(self, params, active_qubits)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        # Draw and Print Circuit:
        drawing = qml.draw(conv_circuit, expansion_strategy="device")(self, params, active_qubits)
        print(drawing)


    # ******* Original Fully Connected Layer Drawing *******:
    def draw_fc_lppc(self, params, active_qubits=None, n_qubits=None,
                     check_qubits=True, version="unitary"):
        """
        Draws the fully connected layer functions for the LPPC QCNN package. Takes in a parameter 
        'version' to denote the version of the fully connected layer that you want (2 total), and 
        defaults to the most recent version if nothing is passed. Uses qml.draw() to complete
        circuit diagrams (Note: argument for training data ('x') removed because it was not accessed,
        as of 7/17/2024).
        """
        # Check Qubit Configuration:
        if check_qubits is True:
            # Active Qubits:
            if active_qubits is None:
                active_qubits = self.active_qubits
                active_qubits = list(range(active_qubits))
            
            # Number of Qubits:
            if n_qubits is None:
                n_qubits = self.n_qubits

        # Initialize Device:
        dev = qml.device("default.qubit", wires=n_qubits)

        # CIRCUIT:
        @qml.qnode(dev)
        def fc_circuit(self, params, active_qubits):
            if version == "original":
                self.fc_layer_orig(self, params, active_qubits)
            elif version == "unitary":
                self.fc_layer_unitary(self, params, active_qubits)
            else:
                raise ValueError("Invalid version. Available versions are 'original' for 'fc_layer_orig' and 'unitary' for 'fc_layer_unitary'.")
            
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        # Draw and Print Circuit:
        drawing = qml.draw(fc_circuit, expansion_strategy="device")(self, params, active_qubits)
        print(drawing)


# ==============================================================================================
#                               OPTIMIZATION AND TRAINING CLASS
# ==============================================================================================


class TrainQC(DrawQC):
    """
    Contains functions for optimization steps in a Quantum Convolutional Neural Network (QCNN).
    It depends on the layer functions provided in the QCircuitLPPC class.
    """
    def __init__(self):
        self.layers = LayersQC()
        self.qc_draw = DrawQC()
        # QUBITS:
        self.n_qubits = 6 # Set 'n_qubits' equal to desired qubit configuration (for us, 6)
        self.n_qubits_draw = 2 # 2 qubit config for DRAWQC
        # ACTIVE QUBITS:
        self.active_qubits = 6 # Set 'active_qubits' equal to 'n_qubits''
        # WIRES:
        self.num_wires = 2
        self.wires = 6

        # TRAINING:
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

    # -----------------------------------------------------------
    #         ORIGINAL OPTIMIZATION FUNCTIONS (LPPC)
    # -----------------------------------------------------------
    
    # ******* Select QCNN Optimizer *******:
    def qcnn_opt_select(self, opt_methods=None, opt_num=None):
        """
        Selects and returns the desired optimizer from the given list of optimizers, based on the 
        given value for 'opt_num'. Allows list of usable optimizers to be appended if necessary, but 
        defaults to a list of optimizers including the Stochastic Gradient Descent (SGD) Optimizer, 
        the ADAM Optimizer, RMS Prop Optimizer, and more (6 total, using PennyLane). 
        Raises "ValueError: If opt_num is not 1, 2, or 3".

        Note: In "qc_opt_list()", "opt_methods" is a list of optimizer options with string versions of
        their associated names, and not actual optimizers themselves. The actual optimizers are 
        instantiated with "opt_methods" in the function "qc_opt_select()".
        """
        # List of Available Optimizers (SIX total):
        if opt_methods is None:
            opt_methods = {
                1: qml.GradientDescentOptimizer(),
                2: qml.AdamOptimizer(),
                3: qml.RMSPropOptimizer(),
                4: qml.MomentumOptimizer(),
                5: qml.NesterovMomentumOptimizer(),
                6: qml.AdagradOptimizer(),
            }
            # Instantiate Default Optimizer (Gradient Descent):
            if opt_num is None:
                # Returns 'GradientDescentOptimizer' if No Selection is Made:
                return qml.GradientDescentOptimizer()

        if opt_num not in opt_methods:
            raise ValueError("opt_num must be equal to an integer value 1-6. for default. Use 'qc_opt_print' to check full list.")

        # Get Optimizer Selection:
        opt = opt_methods.get(opt_num)

        return opt

    # ******* List Available QCNN Optimizers *******:
    def qcnn_opt_list(self):
        """
        Prints the list of optimizer options and their associated 'opt_num' values. Allows list of 
        usable optimizers to be appended if necessary, but defaults to a list of optimizers including
        the Stochastic Gradient Descent (SGD) Optimizer, the ADAM Optimizer, RMS Prop Optimizer, 
        and more (6 total, using PennyLane). 
        
        Note: In "qc_opt_list()", "opt_names" is a list of optimizer options with string versions of
        their associated names, and not actual optimizers themselves. The actual optimizers are 
        instantiated with "opt_methods" in the function "qc_opt_select()".
        """
        # List of potential optimizers (SIX total):
        opt_names = {
            1: "GradientDescentOptimizer",
            2: "AdamOptimizer",
            3: "RMSPropOptimizer",
            4: "MomentumOptimizer",
            5: "NesterovMomentumOptimizer",
            6: "AdagradOptimizer"
        }

        # Print Available Optimizers:
        print("Available Optimizer Options:")
        print("-------")
        for num, name in opt_names.items():
            print(f"{num}: {name}")
            print("-------")

    # ******* Original Mean Squared Error Cost *******:
    def mse_cost(self, params, x, y, n_qubits=None):
        """
        Computes the Mean Squared Error (MSE) cost function (Note: Specifically 
        calculates the MSE for the updated version of the LPPC QCNN V2).
        """
        # Check Number of Qubits:
        if n_qubits is None:
            n_qubits = self.n_qubits

        # Calculate Predictions:
        predictions = np.array([self.qcircuit_lppc(self, params, xi) for xi in x])
        
        return np.mean((predictions - y) ** 2)

    # ******* Original Stochastic Gradient Descent *******:
    def stoch_grad_orig(self, opt, cost, params, x, y, learning_rate, batch_size, max_iterations,
                      conv_tol, active_qubits=None, n_qubits=None, check_qubits=True):
        """
        Updates parameters using stochastic gradient descent and returns the updated parameters
        and average cost (original version).
        """
        # Check Qubit Configuration:
        if check_qubits is True:
            # Active Qubits:
            if active_qubits is None:
                active_qubits = self.active_qubits
                active_qubits = list(range(active_qubits))
            
            # Number of Qubits:
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

            for n in range(max_iterations):
                params, prev_cost = opt.step_and_cost(lambda v: cost(self, v, x_batch, y_batch),
                                                      params)

                # Compute Cost for Current Params:
                batch_cost = cost(self, params, x_batch, y_batch)
                total_cost += batch_cost * len(x_batch)  # Accumulate Total Cost

                # Convergence Check:
                conv = np.abs(batch_cost - prev_cost)
                if n % 10 == 0:
                    print(f"Step {n}: Cost function = {batch_cost:.8f}")

                if conv <= conv_tol:
                    break

        # Average Total Cost over All Samples:
        avg_cost = total_cost / len(x)
        
        return params, avg_cost

    # ******* Original Accuracy *******:
    def accuracy_orig(self, predictions, y):
        """
        Calculates the accuracy of the QCNN model on the provided testing data. Assumes
        predictions were calculated already, not dependent on quantum circuit
        function (original version).
        """
        # Calculate Number of Correct Predictions:
        true_predictions = np.sum(predictions == y)

        # Calculate Accuracy:
        accuracy = true_predictions / len(y)

        return accuracy
    
    # -----------------------------------------------------------
    #               NEW OPTIMIZATION FUNCTIONS
    # -----------------------------------------------------------

    # ******* TO-DO *******:
    def todo():
        """
        FILL.
        """
        # TO-DO

        return None


# **********************************************************************************************
#                                          MAIN
# **********************************************************************************************


def main():
    return


if __name__ == "__main__":
    main()
