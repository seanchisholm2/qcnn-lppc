########################################## qcircuit.py ############################################

### IMPORTS / DEPENDENCIES:

# PennyLane:
import pennylane as qml
from pennylane import numpy as np
# from pennylane.templates import RandomLayers  # NOT ACCESSED

# Data/Modeling:
# from sklearn import datasets  # NOT ACCESSED
# from sklearn.model_selection import train_test_split  # NOT ACCESSED
# from sklearn.svm import SVC  # NOT ACCESSED
# from sklearn.preprocessing import OneHotEncoder  # NOT ACCESSED

# Plotting:
# import matplotlib as mpl  # NOT ACCESSED
# mpl.rcParams.update(mpl.rcParamsDefault)  # NOT ACCESSED
# import matplotlib.pyplot as plt  # NOT ACCESSED


################### QUANTUM CIRCUIT AND LAYERS CLASS ######################


# CLASS IMPORTS ("gellmann_ops.py"):
from .gellmann_ops import GellMannOps as gell_ops

class QCircuitLPPC:
    """
    Contains functions for different layers in a quantum convolutional neural network and
    defines quantum circuits.
    """
    def __init__(self):
        self.gell_ops = gell_ops() # Initialize 'GellMannOps' to access variables
        self.n_qubits = self.gell_ops.n_qubits
        self.num_active_qubits = self.gell_ops.num_active_qubits
        self.num_qubits = self.gell_ops.num_qubits
        self.active_qubits = self.gell_ops.active_qubits

    # CONVOLUTIONAL LAYER (VERSION #1):
    def conv_layer_V1(self, weights, active_qubits,
                      n_qubits=None):
        """
        Applies a quantum convolutional layer (VERSION #1).
        """
        # QUBIT CHECK:
        #-------------------------------------
        # Check 'n_qubits' is passed:
        if n_qubits is None:
            # n_qubits = self.n_qubits
            # n_qubits = 10
            n_qubits = 2 # FOR TESTING
        #-------------------------------------

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


    # CONVOLUTIONAL LAYER (VERSION #2; CURRENT VERSION):
    def conv_layer_V2(self, weights, active_qubits,
                      n_qubits=None):
        """
        Applies a quantum convolutional layer (VERSION #2).
        """
        # QUBIT CHECK:
        #-------------------------------------
        # Check 'n_qubits' is passed:
        if n_qubits is None:
            # n_qubits = self.n_qubits
            # n_qubits = 10
            n_qubits = 2 # FOR TESTING
        #-------------------------------------

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


    # POOLING LAYER (VERSION #1):
    def pool_layer_V1(self, weights, active_qubits,
                      n_qubits=None):
        """
        Applies two-qubit pooling operation to inputted qubits, including controlled rotation 
        based on a conditional measurement (VERSION #1).
        """
        # QUBIT CHECK:
        #-------------------------------------
        # Check 'n_qubits' is passed:
        if n_qubits is None:
            # n_qubits = self.n_qubits
            # n_qubits = 10
            n_qubits = 2 # FOR TESTING
        #-------------------------------------

        # Generate Gell-Mann matrices for 2-D space (single qubit operators):
        pool_operators = gell_ops.generate_gell_mann(gell_ops, (2**n_qubits))

        # Loop over all active qubits (in pairs):
        for i in range(0, self.n_qubits, 2):
            q1 = active_qubits[i]   # First qubit
            q2 = active_qubits[i+1] # Second qubit

            # Get convolutional operators V1 and V2 from pool operators and weights:
            # FIRST OPERATOR:
            v1 = gell_ops.get_conv_op(gell_ops, pool_operators,
                                           weights[i])  # V1 -> First set of weights
            # SECOND OPERATOR:
            v2 = gell_ops.get_conv_op(gell_ops, pool_operators,
                                           weights[i+1])  # V2 -> Second set of weights

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
        Applies two-qubit pooling operation to inputted qubits, including controlled rotation 
        based on a conditional measurement (VERSION #2).
        """
        # QUBIT CHECK:
        #-------------------------------------
        # Check 'n_qubits' is passed:
        if n_qubits is None:
            # n_qubits = self.n_qubits
            # n_qubits = 10
            n_qubits = 2 # FOR TESTING
        #-------------------------------------

        # Generate Gell-Mann matrices for 2-D space (single qubit operators):
        pool_operators = gell_ops.generate_gell_mann(gell_ops, (2**n_qubits))

        # Ensure active_qubits is *list* of qubit indices:
        if isinstance(active_qubits, int):
            active_qubits = list(range(active_qubits))

        # Loop over all active qubits (in pairs):
        index = 0
        while index + 2 < len(active_qubits):
            q1 = active_qubits[index]   # First qubit
            q2 = active_qubits[index+1] # Second qubit

            # Get convolutional operators V1 and V2 from pool operators and weights:
            v1 = gell_ops.get_conv_op(gell_ops, pool_operators,
                                           weights[index])  # V1 -> First set of weights
            v2 = gell_ops.get_conv_op(gell_ops, pool_operators,
                                           weights[index+1])  # V2 -> Second set of weights

            # Apply Hadamard Gate to first qubit:
            qml.Hadamard(wires=q1)

            # Apply first Controlled Unitary Operation:
            qml.ControlledQubitUnitary(v1, control_wires=[q1], target_wires=[q2])

            # Perform PauliZ expectation value measurement on second qubit:
            # qml.measure(q2)

            index += 2

        # Update active qubits by pooling 1 out of every 2 qubits:
        self.active_qubits = active_qubits[::2]


    # POOLING LAYER (VERSION #3; CURRENT VERSION):
    def pool_layer_V3(self, weights, active_qubits,
                      n_qubits=None):
        """
        Applies two-qubit pooling operation to inputted qubits, including controlled rotation 
        based on a conditional measurement (VERSION #3).
        """
        # QUBIT CHECK:
        #-------------------------------------
        # Check 'n_qubits' is passed:
        if n_qubits is None:
            # n_qubits = self.n_qubits
            # n_qubits = 10
            n_qubits = 2 # FOR TESTING
        #-------------------------------------
        
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


    # FULLY CONNECTED LAYER (VERSION #1):
    def fully_connected_layer_V1(self, params, active_qubits,
                                 n_qubits=None):
        """
        Applies a fully connected layer to the remaining active qubits (VERSION #1).
        """
        # QUBIT CHECK:
        #-------------------------------------
        # Check 'n_qubits' is passed:
        if n_qubits is None:
            # n_qubits = self.n_qubits
            # n_qubits = 10
            n_qubits = 2 # FOR TESTING
        #-------------------------------------

        num_qubits = len(active_qubits)  # Initialize active qubits

        # Apply Fully Connected Operator to active qubits:
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                qml.CNOT(wires=[active_qubits[i], active_qubits[j]])
                qml.RY(params[i][j], wires=active_qubits[j])
                qml.CNOT(wires=[active_qubits[i], active_qubits[j]])


    # FULLY CONNECTED LAYER (VERSION #2; CURRENT VERSION):
    def fully_connected_layer_V2(self, params, active_qubits,
                                 n_qubits=None):
        """
        Applies a fully connected layer to the remaining active qubits (VERSION #2).
        """
        # QUBIT CHECK:
        #-------------------------------------
        # Check 'n_qubits' is passed:
        if n_qubits is None:
            # n_qubits = self.n_qubits
            # n_qubits = 10
            n_qubits = 2 # FOR TESTING
        #-------------------------------------
        
        # Generate Gell-Mann matrices for vector space:
        fc_mats = gell_ops.generate_gell_mann(gell_ops, 2**len(n_qubits))
        fc_op = gell_ops.get_conv_op(gell_ops, fc_mats, params)

        # Apply Fully Connected Operator to active qubits:
        qml.QubitUnitary(fc_op, wires=active_qubits)


    # QUANTUM CIRCUIT (VERSION #1):
    def qcircuit_V1(self, params, x, active_qubits=None, n_qubits=None):
        """
        Defines a quantum circuit for a Quantum Convolutional Neural Network (QCNN). Encodes
        input features using amplitude embedding, applies a convolutional layer, a pooling layer,
        and a fully connected layer, and then measures the qubits (VERSION #1).
        """
        # ACTIVE QUBIT CHECK:
        #-------------------------------------------------
        # Check 'active_qubits' is passed:
        if active_qubits is None:
            # active_qubits = self.active_qubits
            # active_qubits = 10
            active_qubits = 2 # FOR TESTING
            active_qubits = list(range(active_qubits))
        
        # Check 'n_qubits' is passed:
        if n_qubits is None:
            # n_qubits = self.n_qubits
            # n_qubits = 10
            n_qubits = 2 # FOR TESTING
        #-------------------------------------------------

        # Apply Amplitude Embedding:
        qml.AmplitudeEmbedding(x, wires=range(self.n_qubits), normalize=True)

        # Apply Convolutional Layer (pass 'params' as argument):
        self.conv_layer_V1(params, active_qubits)

        # Apply Pooling Layer (pass 'params' as argument):
        self.pool_layer_V3(params, active_qubits)

        # Apply Fully Connected Layer (pass 'params' as argument):
        self.fully_connected_layer_V2(params, active_qubits)

        # Measure remaining Active Qubits:
        return qml.expval(qml.PauliZ(active_qubits))


    # QUANTUM CIRCUIT (VERSION #2; CURRENT VERSION):
    def qcircuit_V2(self, params, x, active_qubits=None, n_qubits=None,
                    draw=False):
        """
        Defines a quantum circuit for a Quantum Convolutional Neural Network (QCNN). Encodes
        input features using amplitude embedding, applies a convolutional layer, a pooling layer,
        and a fully connected layer, and then measures the qubits (VERSION #2).
        """
        # ACTIVE QUBIT CHECK:
        #-------------------------------------------------
        # Check 'active_qubits' is passed:
        if active_qubits is None:
            # active_qubits = self.active_qubits
            # active_qubits = 10
            active_qubits = 2 # FOR TESTING
            active_qubits = list(range(active_qubits))
        
        # Check 'n_qubits' is passed:
        if n_qubits is None:
            # n_qubits = self.n_qubits
            # n_qubits = 10
            n_qubits = 2 # FOR TESTING
        #-------------------------------------------------

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
    

    # ******* UPDATED LAYER VERSION(S) *******

    # UPDATED CONVOLUTIONAL LAYER FUNCTION:
    def conv_layer_lppc(self, *args, **kwargs):
        """
        Returns the most recent version of the CONVOLUTIONAL LAYER used in the 
        QCNN (CURRENT VERSION: V2).
        """
        # Return Current Convolutional Layer ('conv_layer_V2') with appropriate arguments:
        return self.conv_layer_V2(self, *args, **kwargs)


    # UPDATED POOLING LAYER FUNCTION:
    def pool_layer_lppc(self, *args, **kwargs):
        """
        Returns the most recent version of the POOLING LAYER used in the 
        QCNN (CURRENT VERSION: V3).
        """
        # Return Current Pooling Layer ('pool_layer_V3') with appropriate arguments:
        return self.pool_layer_V3(self, *args, **kwargs)

    
    # UPDATED FULLY CONNECTED LAYER FUNCTION:
    def fc_layer_lppc(self, *args, **kwargs):
        """
        Returns the most recent version of the FULLY CONNECTED LAYER used in the
        QCNN (CURRENT VERSION: V2).
        """
        # Return Current Fully Connected C Layer ('fully_connected_layer_V2') with 
        # appropriate arguments:
        return self.fully_connected_layer_V2(self, *args, **kwargs)

        
    # UPDATED QUANTUM CIRCUIT FUNCTION:
    def qcircuit_lppc(self, *args, **kwargs):
        """
        Returns the most recent version of the QUANTUM CIRCUIT used in the
        QCNN (CURRENT VERSION: V2).
        """
        # Return Current Quantum Circuit Function ('qcircuit_V2') with
        # appropriate arguments:
        return self.qcircuit_V2(self, *args, **kwargs)  


################### QCNN DRAWING CLASS ######################


class DrawQC(QCircuitLPPC):
    """
    Contains functions for drawing different layer functions relevant to the
    quantum convolutional neural network, including convolutional layers, pooling layers,
    fully connected layers, and quantum circuits. Defaults to a 2-Qubit representation of
    each layer, as well as defaulting to the most recent version of each layer function if
    none is specified.
    """
    def __init__(self):
        super().__init__()
        self.gell_ops = gell_ops() # Initialize 'GellMannOps' to access variables
        self.qc_circ = QCircuitLPPC() # Initialize 'QCircuitLPPC' to access circuit functions
        self.n_qubits = self.gell_ops.n_qubits
        self.num_active_qubits = self.gell_ops.num_active_qubits
        self.num_qubits = self.gell_ops.num_qubits
        self.active_qubits = self.gell_ops.active_qubits

    # POOLING LAYER DRAWING FUNCTION:
    def draw_pool_layer(self, params, x, active_qubits=None, n_qubits=None,
                        version=None):
        """
        Draws the pooling layer functions for the LPPC QCNN package. Takes in a parameter 
        'version' to denote the version of the pooling layer that you want (3 total), and 
        defaults to the most recent version if nothing is passed. Uses qml.draw() to complete
        circuit diagrams.
        """
        # ACTIVE QUBIT CHECK:
        #-------------------------------------------------
        # Check 'active_qubits' is passed:
        if active_qubits is None:
            # active_qubits = self.active_qubits
            # active_qubits = 10
            active_qubits = 2 # FOR TESTING
            active_qubits = list(range(active_qubits))
        
        # Check 'n_qubits' is passed:
        if n_qubits is None:
            # n_qubits = self.n_qubits
            # n_qubits = 10
            n_qubits = 2 # FOR TESTING
        #-------------------------------------------------
        
        # Default Recent Version (V3):
        num_pool = [1, 2, 3] # Initialize Number of Versions

        # VERSION CHECK:
        if version is None:
            version = num_pool[-1]
        if version not in num_pool:
            raise ValueError("Version of pooling layer must equal 1, 2, or 3.")

        # Circuit to Draw Layer:
        #------------------------------------------------------------------------
        dev = qml.device("default.qubit", wires=n_qubits) # Initialize Device

        # Initialize Circuit:
        @qml.qnode(dev)
        def pool_circuit(self, params, active_qubits):
            if version == 1:
                self.pool_layer_V1(self, params, active_qubits)
            elif version == 2:
                self.pool_layer_V2(self, params, active_qubits)
            elif version == 3:
                self.pool_layer_V3(self, params, active_qubits)
            
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        #------------------------------------------------------------------------

        # Draw and Print Circuit:
        drawing = qml.draw(pool_circuit, expansion_strategy="device")(self, params, active_qubits)
        print(drawing)


    # CONVOLUTIONAL LAYER DRAWING FUNCTION:
    def draw_conv_layer(self, params, x, active_qubits=None, n_qubits=None,
                        version=None):
        """
        Draws the convolutional layer functions for the LPPC QCNN package. Takes in a parameter 
        'version' to denote the version of the convolutional layer that you want (2 total), and 
        defaults to the most recent version if nothing is passed. Uses qml.draw() to complete
        circuit diagrams.
        """
        # ACTIVE QUBIT CHECK:
        #-------------------------------------------------
        # Check 'active_qubits' is passed:
        if active_qubits is None:
            # active_qubits = self.active_qubits
            # active_qubits = 10
            active_qubits = 2 # FOR TESTING
            active_qubits = list(range(active_qubits))
        
        # Check 'n_qubits' is passed:
        if n_qubits is None:
            # n_qubits = self.n_qubits
            # n_qubits = 10
            n_qubits = 2 # FOR TESTING
        #-------------------------------------------------

        # Default Recent Version (V2):
        num_conv = [1, 2] # Initialize Number of Versions
        
        # VERSION CHECK:
        if version is None:
            version = num_conv[-1]
        if version not in num_conv:
            raise ValueError("Version of convolutional layer must equal 1 or 2.")

        # Circuit to Draw Layer:
        #------------------------------------------------------------------------
        dev = qml.device("default.qubit", wires=n_qubits) # Initialize Device

        # Initialize Circuit:
        @qml.qnode(dev)
        def conv_circuit(self, params, active_qubits):
            if version == 1:
                self.conv_layer_V1(self, params, active_qubits)
            elif version == 2:
                self.conv_layer_V2(self, params, active_qubits)
            
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        #------------------------------------------------------------------------

        # Draw and Print Circuit:
        drawing = qml.draw(conv_circuit, expansion_strategy="device")(self, params, active_qubits)
        print(drawing)


    # FULLY CONNECTED LAYER DRAWING FUNCTION:
    def draw_fc_layer(self, params, x, active_qubits=None, n_qubits=None,
                      version=None):
        """
        Draws the fully connected layer functions for the LPPC QCNN package. Takes in a parameter 
        'version' to denote the version of the fully connected layer that you want (2 total), and 
        defaults to the most recent version if nothing is passed. Uses qml.draw() to complete
        circuit diagrams.
        """
        # ACTIVE QUBIT CHECK:
        #-------------------------------------------------
        # Check 'active_qubits' is passed:
        if active_qubits is None:
            # active_qubits = self.active_qubits
            # active_qubits = 10
            active_qubits = 2 # FOR TESTING
            active_qubits = list(range(active_qubits))
        
        # Check 'n_qubits' is passed:
        if n_qubits is None:
            # n_qubits = self.n_qubits
            # n_qubits = 10
            n_qubits = 2 # FOR TESTING
        #-------------------------------------------------

        # Default Recent Version (V2):
        num_fc = [1, 2] # Initialize Number of Versions
        
        # VERSION CHECK:
        if version is None:
            version = num_fc[-1]
        if version not in num_fc:
            raise ValueError("Version of fuly connected layer must equal 1 or 2.")

        # Circuit to Draw Layer:
        #------------------------------------------------------------------------
        dev = qml.device("default.qubit", wires=n_qubits) # Initialize Device

        # Initialize Circuit:
        @qml.qnode(dev)
        def fc_circuit(self, params, active_qubits):
            if version == 1:
                self.fully_connected_layer_V1(self, params, active_qubits)
            elif version == 2:
                self.fully_connected_layer_V2(self, params, active_qubits)
            
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        #------------------------------------------------------------------------

        # Draw and Print Circuit:
        drawing = qml.draw(fc_circuit, expansion_strategy="device")(self, params, active_qubits)
        print(drawing)


################### OPTIMIZATION AND COST CLASS ######################


class OptStepLPPC(QCircuitLPPC):
    """
    Contains functions for optimization steps in a Quantum Convolutional Neural Network (QCNN).
    It depends on the layer functions provided in the QCircuitLPPC class.
    """
    def __init__(self):
        super().__init__()
        self.qc_circ = QCircuitLPPC() # Initialize 'QCircuitLPPC' to access circuit functions
        self.learning_rate = 0.01
        self.num_steps = 100
        self.batch_size = 10
        self.max_iteration = 100
        self.conv_tol = 1e-06

        self.gell_ops = gell_ops() # Initialize GellMannOps to access variables
        self.n_qubits = self.gell_ops.n_qubits
        self.num_active_qubits = self.gell_ops.num_active_qubits
        self.num_qubits = self.gell_ops.num_qubits
        self.active_qubits = self.gell_ops.active_qubits

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

    # SELECT OPTIMIZER:
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


    # LIST AVAILABLE OPTIMIZERS:
    def qcnn_opt_list(self, opt_names=None):
        """
        Prints the list of optimizer options and their associated 'opt_num' values. Allows list of 
        usable optimizers to be appended if necessary, but defaults to a list of optimizers including
        the Stochastic Gradient Descent (SGD) Optimizer, the ADAM Optimizer, RMS Prop Optimizer, 
        and more (6 total, using PennyLane). 
        
        Note: In "qc_opt_list()", "opt_names" is a list of optimizer options with string versions of
        their associated names, and not actual optimizers themselves. The actual optimizers are 
        instantiated with "opt_methods" in the function "qc_opt_select()".
        """
        # Check for Optimizer List, Assign Default if None:
        if opt_names is None:
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


    # MEAN SQUARED ERROR (MSE) COST FUNCTION:
    def mse_cost(self, params, x, y, n_qubits=None):
        """
        Computes the Mean Squared Error (MSE) cost function (Note: Specifically 
        calculates the MSE for the updated version of the LPPC QCNN V2).
        """
        # QUBIT CHECK:
        #-------------------------------------
        # Check 'n_qubits' is passed:
        if n_qubits is None:
            # n_qubits = self.n_qubits
            # n_qubits = 10
            n_qubits = 2 # FOR TESTING
        #-------------------------------------

        # Calculate Predictions:
        predictions = np.array([self.qcircuit_lppc(self, params, xi) for xi in x])
        
        return np.mean((predictions - y) ** 2)


    # STOCHASTIC GRADIENT DESCENT FUNCTION (VERSION #1):
    def stoch_grad_V1(self, params, x, y, learning_rate, batch_size,
                              active_qubits=None, n_qubits=None):
        """
        Updates parameters using stochastic gradient descent and returns the updated 
        parameters and average cost (VERSION #1).
        """
        # ACTIVE QUBIT CHECK:
        #-------------------------------------------------
        # Check 'active_qubits' is passed:
        if active_qubits is None:
            # active_qubits = self.active_qubits
            # active_qubits = 10
            active_qubits = 2 # FOR TESTING
            active_qubits = list(range(active_qubits))
        
        # Check 'n_qubits' is passed:
        if n_qubits is None:
            # n_qubits = self.n_qubits
            # n_qubits = 10
            n_qubits = 2 # FOR TESTING
        #-------------------------------------------------

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

        # Average Total Cost over All Samples:
        avg_cost = total_cost / len(x)

        return params, avg_cost

    
    # STOCHASTIC GRADIENT DESCENT FUNCTION (VERSION #2):
    def stoch_grad_V2(self, opt, cost, params, x, y, learning_rate, batch_size, max_iterations,
                      conv_tol, active_qubits=None, n_qubits=None):
        """
        Updates parameters using stochastic gradient descent and returns the updated parameters
        and average cost (VERSION #2).
        """
        # ACTIVE QUBIT CHECK:
        #-------------------------------------------------
        # Check 'active_qubits' is passed:
        if active_qubits is None:
            # active_qubits = self.active_qubits
            # active_qubits = 10
            active_qubits = 2 # FOR TESTING
            active_qubits = list(range(active_qubits))
        
        # Check 'n_qubits' is passed:
        if n_qubits is None:
            # n_qubits = self.n_qubits
            # n_qubits = 10
            n_qubits = 2 # FOR TESTING
        #-------------------------------------------------
        
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


    # ACCURACY (VERSION #1):
    def accuracy_V1(self, predictions, y):
        """
        Calculates the accuracy of the QCNN model on the provided testing data. Assumes
        predictions were calculated already, not dependent on quantum circuit
        function (VERSION #1).
        """
        # Calculate Number of Correct Predictions:
        true_predictions = np.sum(predictions == y)

        # Calculate Accuracy:
        accuracy = true_predictions / len(y)

        return accuracy
    
    # ACCURACY (VERSION #2):
    def accuracy_V2(self, predictions, y):
        """
        Calculates the accuracy and precision of the QCNN model on the provided testing 
        data. Assumes predictions were calculated already, not dependent on quantum 
        circuit function. Includes checks for data type consistency, matching shapes,
        and handling empty arrays (VERSION #2).
        """
        # Ensure Data Type Consistency:
        predictions = np.asarray(predictions)
        y = np.asarray(y)

        # Check 'predictions' and 'y' have Same Shape:
        if predictions.shape != y.shape:
            raise ValueError("Shape of predictions and y must match.")

        # Handle Empty Arrays:
        if len(y) == 0:
            raise ValueError("The array of true labels 'y' is empty.")

        # Calculate Number of Correct Predictions:
        true_predictions = np.sum(predictions == y)

        # Calculate Accuracy:
        accuracy = true_predictions / len(y)

        # Calculate Precision:
        true_positives = np.sum((predictions == 1) & (y == 1))
        predicted_positives = np.sum(predictions == 1)
        # Adding Epsilon for Numerical Stability:
        precision = true_positives / (predicted_positives + np.finfo(float).eps)

        return accuracy, precision


    # ******* UPDATED OPTIMIZATION VERSION(S) *******


    # UPDATED STOCHASTIC GRADIENT DESCENT FUNCTION:
    def stoch_grad_lppc(self, *args, **kwargs):
        """
        Returns the most recent version of the STOCHASTIC GRADIENT DESCENT optimization
        function used in the QCNN (CURRENT VERSION: V2).
        """
        # Return Current Stochastic Gradient Descent Function ('stoch_grad_V2') with
        # appropriate arguments:
        return self.stoch_grad_V2(self, *args, **kwargs)
    
    # UPDATED ACCURACY FUNCTION:
    def accuracy_lppc(self, *args, **kwargs):
        """
        Returns the most recent version of the ACCURACY function used in the 
        QCNN (CURRENT VERSION: V1).
        """
        # Return Current Accuracy Function ('accuracy_V1') with appropriate arguments:
        return self.accuracy_V1(self, *args, **kwargs)


################### MAIN ######################


def main():
    return


if __name__ == "__main__":
    main()
