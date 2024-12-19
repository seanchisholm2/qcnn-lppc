########################################### CIRCUIT_LAYERS.PY ###########################################

#### ***** IMPORTS / DEPENDENCIES *****:

### *** PLOTTING ***:
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use('./qcnn-figures/chisholm-lppc.mplstyle')

### *** PENNYLANE ***:
import pennylane as qml
import pennylane.numpy as pnp

### *** DATA ***:
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()

### *** JAX ***:
import jax;
## JAX CONFIGURATIONS:
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
## OTHER (JAX):
import optax

### *** OTHER ***:
from datetime import datetime # labeling images

### *** RNG ***:
seed = 0

### ***** PACKAGE(S) *****:
# **********************************************************************************
# *1* FROM OPERATORS.PY:
from .qc_operators import QuantumMathOps as qmath_ops # (STATIC METHOD)
# *1* FROM LOAD_QC_DATA.PY:
from .load_qc_data import LoadDataQC # (STATIC METHOD)
from .load_qc_data import LoadPhotonData # (STATIC METHOD)
# **********************************************************************************


# ============================================================
#                    NEW QCNN LAYERS CLASS                    
# ============================================================


class LayersQC:
    """
    Contains all relevant circuit and layer functions for the quantum convolutional neural
    network (LPPC).
    """
    ## DEVICE
    # *1* Define number of wires for device (easier to switch between device types):
    device_wires = 6
    # *2* Select device for class:
    device = qml.device('default.qubit.jax', wires=device_wires)
    # device = qml.device("default.qubit", wires=device_wires)
    # device = qml.device("default.mixed", wires=device_wires)

    def __init__(self):
        ## CLASSES:
        self.qmo = qmath_ops
        ## QUBITS:
        self.n_qubits = 6 # Set 'n_qubits' equal to desired qubit configuration (for us, 6)
        self.n_qubits_draw = 2 # 2 qubit config for DRAWQC
        ## ACTIVE QUBITS:
        self.active_qubits = 6 # Set 'active_qubits' equal to 'n_qubits''
        ## WIRES:
        self.wires = 6
        self.n_wires = 6
        self.num_wires = 6

    # ----------------------------------------------------
    #     CIRCUIT AND LAYER FUNCTIONS (NEW/ESSENTIAL)
    # ----------------------------------------------------

    # ******* CONVOLUTIONAL LAYER *******:
    def convolutional_layer(self, weights, wires, skip_first_layer=True):
        """
        Adds a convolutional layer to a circuit.

        Args:
            -> weights (np.array): 1D array with 15 weights of the parametrized gates.
            -> wires (list[int]): Wires where the convolutional layer acts on.
            -> skip_first_layer (bool): Skips the first two U3 gates of a layer.
        """
        # Check Wires:
        if wires is None:
            wires = self.wires
        
        # Check Number of Wires:
        n_wires = len(wires)
        assert n_wires >= 3, "this circuit is too small!"

        for p in [0, 1]:
            for indx, w in enumerate(wires):
                if indx % 2 == p and indx < n_wires - 1:
                    if indx % 2 == 0 and not skip_first_layer:
                        qml.U3(*weights[:3], wires=[w])
                        qml.U3(*weights[3:6], wires=[wires[indx + 1]])
                    qml.IsingXX(weights[6], wires=[w, wires[indx + 1]])
                    qml.IsingYY(weights[7], wires=[w, wires[indx + 1]])
                    qml.IsingZZ(weights[8], wires=[w, wires[indx + 1]])
                    qml.U3(*weights[9:12], wires=[w])
                    qml.U3(*weights[12:], wires=[wires[indx + 1]])

    # ******* POOLING LAYER *******:
    def pooling_layer(self, weights, wires):
        """
        Adds a pooling layer to a circuit.

        Args:
            -> weights (np.array): Array with the weights of the conditional U3 gate.
            -> wires (list[int]): List of wires to apply the pooling layer on.
        """
        # Check Wires:
        if wires is None:
            wires = self.wires
        
        # Check Number of Wires:
        n_wires = len(wires)
        assert len(wires) >= 2, "this circuit is too small!"

        for indx, w in enumerate(wires):
            if indx % 2 == 1 and indx < n_wires:
                m_outcome = qml.measure(w)

                qml.cond(m_outcome, qml.U3)(*weights, wires=wires[indx - 1])

    # ******* FOUR-QUBIT UNITARY CONVOLUTIONAL LAYER *******:
    def four_conv_layer(self, params, active_qubits, barrier=True):
        """
        Adds a four-qubit unitary convolutional layer to a circuit.

        Args:
            -> params (np.array): Array with the parameters for the convolutional operations.
            -> active_qubits (list[int]): List of active qubits to apply the convolutional layer on.
            -> barrier (bool): Whether to add a barrier after the operations.
        """
        conv_operators = self.qmo.generate_gell_mann(self.qmo, 4)  # Two-qubit gell mann matricies
        u_conv = self.qmo.get_conv_op(self.qmo, conv_operators, params)  

        start_index = 0 
        index = start_index

        while index + 3 < len(active_qubits):
            q_index = active_qubits[index]
            q_index_1 = active_qubits[index + 1]
            q_index_2 = active_qubits[index + 2]
            q_index_3 = active_qubits[index + 3]


            qml.QubitUnitary(u_conv, [q_index, q_index_1])
            qml.QubitUnitary(u_conv, [q_index, q_index_3])
            qml.QubitUnitary(u_conv, [q_index, q_index_2])
            qml.QubitUnitary(u_conv, [q_index_1, q_index_3])
            qml.QubitUnitary(u_conv, [q_index_1, q_index_2])
            qml.QubitUnitary(u_conv, [q_index_2, q_index_3])
            
            qml.Barrier()

            if index == 0:
                index += 2
            else:
                index += 3

        if barrier:
            qml.Barrier()

    # ******* THREE-QUBIT UNITARY CONVOLUTIONAL LAYER *******:
    def three_conv_layer(self, params, active_qubits, barrier=True):
        """
        Adds a three-qubit unitary convolutional layer to a circuit.

        Args:
            -> params (np.array): Array with the parameters for the convolutional operations.
            -> active_qubits (list[int]): List of active qubits to apply the convolutional layer on.
            -> barrier (bool): Whether to add a barrier after the operations.
        """
        conv_operators = self.qmo.generate_gell_mann(self.qmo, 8)  # Three-qubit operators
        u_conv = self.qmo.get_conv_op(self.qmo, conv_operators, params) # params = 63 (?)

        start_index = 0 
        index = start_index

        while index + 2 < len(active_qubits):
            q_index = active_qubits[index]
            q_index_1 = active_qubits[index + 1]
            q_index_2 = active_qubits[index + 2]

            qml.QubitUnitary(u_conv, [q_index, q_index_1, q_index_2])
            index += 3

        if barrier:
            qml.Barrier()

    # ******* CUSTOM CONVOLUTIONAL LAYER *******:
    def custom_conv_layer(self, params, active_qubits, barrier=True):

        start_index = 0 
        index = start_index
        # (Three-qubit convolutions)
        group_size = 2

        while index + (group_size - 1) < len(active_qubits):
            param_pointer = 0
            lst_indicies = range(index, index + group_size)
                # Ascending loop -> z,y
            for axis in ['z', 'y']:
                split_index = group_size - 1
                while split_index > 0:
                    control_indicies = lst_indicies[:split_index]
                    control_qubit_indicies = [active_qubits[i] for i in control_indicies]
                    target_qubit_index = active_qubits[lst_indicies[split_index]]

                    num_local_params = 2**(len(control_qubit_indicies))
                    local_params = params[param_pointer:param_pointer + num_local_params]
                    param_pointer += num_local_params

                    self.qmo.generate_uniformly_controlled_rotation(self.qmo, local_params,
                                                    control_qubit_indicies, target_qubit_index, axis=axis)

                    split_index -= 1

                if axis == 'z':
                    qml.RZ(params[param_pointer], active_qubits[lst_indicies[split_index]])
                else:
                    qml.RY(params[param_pointer], active_qubits[lst_indicies[split_index]])
                param_pointer += 1

            # Descending loop:
            for axis in ['y', 'z']:
                split_index = 1

                if axis == 'z':
                    qml.RZ(params[param_pointer], active_qubits[lst_indicies[split_index-1]])
                    param_pointer += 1

                while split_index < group_size:
                    control_indicies = lst_indicies[:split_index]
                    control_qubit_indicies = [active_qubits[i] for i in control_indicies]
                    target_qubit_index = active_qubits[lst_indicies[split_index]]

                    num_local_params = 2**(len(control_qubit_indicies))
                    local_params = params[param_pointer:param_pointer + num_local_params]
                    param_pointer += num_local_params

                    # (Q: Where does kwarg 'label=label' come from? Removed for now)
                    self.qmo.generate_uniformly_controlled_rotation(self.qmo, local_params,
                                                    control_qubit_indicies, target_qubit_index, axis=axis)

                    split_index += 1

            index += group_size

        if barrier:
            qml.Barrier()

    # ******* CONVOLUTIONAL AND POOLING LAYER *******:
    def conv_and_pooling(self, kernel_weights, n_wires, skip_first_layer=True):
        """
        Applies both the convolutional and pooling layer.
        """
        # Check Number of Wires ('n_wires'):
        if n_wires is None:
            wires = self.n_wires

        if skip_first_layer: 
            self.three_conv_layer(kernel_weights[15:78], n_wires, barrier=True)  
        else: 
            self.four_conv_layer(kernel_weights[:15], n_wires, barrier=True)
            self.three_conv_layer(kernel_weights[15:78], n_wires, barrier=True)

        self.pooling_layer(kernel_weights[78:], n_wires)

    # ******* DENSE LAYER *******:
    def dense_layer(self, weights, wires):
        """
        Applies an arbitrary unitary gate to a specified set of wires.
        """
        # Check Wires:
        if wires is None:
            wires = self.wires

        # Generate Gell-Mann matrices for vector space:
        fc_mats = self.qmo.generate_gell_mann(self.qmo, (2**len(wires)))
        fc_op = self.qmo.get_conv_op(self.qmo, fc_mats, weights)

        # Apply Fully Connected Operator to active qubits:
        qml.QubitUnitary(fc_op, wires=wires)

    # ----------------------------------------------------
    #          CIRCUIT FUNCTIONS (QC AND LAYERS)
    # ----------------------------------------------------

    # ******* QUANTUM CIRCUIT *******:
    @qml.qnode(device, interface='jax')
    def conv_net(self, weights, last_layer_weights, features):
        """
        Defines the QCNN circuit.
        
        Args:
            weights (array): Parameters of the convolution and pool layers.
            last_layer_weights (array): Parameters of the last dense layer.
            features (array): Input data to be embedded using AmplitudeEmbedding.
        """

        layers = weights.shape[1]
        wires = list(range(self.num_wires))

        # inputs the state input_state
        qml.AmplitudeEmbedding(features=features, wires=wires, pad_with=0.5)
        qml.Barrier(wires=wires, only_visual=True)

        # adds convolutional and pooling layers
        for j in range(layers):
            self.conv_and_pooling(weights[:, j], wires, skip_first_layer=(not j == 0))
            wires = wires[::2]
            qml.Barrier(wires=wires, only_visual=True)

        assert last_layer_weights.size == 4 ** (len(wires)) - 1, (
            "The size of the last layer weights vector is incorrect!"
            f" \n Expected {4 ** (len(wires)) - 1}, Given {last_layer_weights.size}"
        )
        self.dense_layer(last_layer_weights, wires)
        return qml.probs(wires=(0))
    
    # ******* THREE-QUBIT UNITARY CONVOLUTIONAL LAYER CIRCUIT *******:
    @qml.qnode(device)
    def three_layer_conv_circuit(self, params, active_qubits):
        """
        Defines a circuit for the three-qubit unitary convolutional layer (for priority use in 
        drawings).
        """
        self.three_conv_layer(params, active_qubits)
        return qml.probs(wires=(0))
    
    # ******* CONVOLUTIONAL AND POOLING LAYER CIRCUIT *******:
    @qml.qnode(device)
    def conv_and_pooling_circuit(self, kernel_weights, n_wires):
        """
        Defines a circuit for the convolutional and pooling layer used in QCNN (for priority use
        in drawings).
        """
        self.conv_and_pooling(kernel_weights, n_wires)
        return qml.probs(wires=(0))


# ============================================================
#            ORIGINAL NEW QCNN LAYERS CLASS (LPPC)
# ============================================================


class LayersLPPC:
    """
    Contains all relevant circuit and layer functions for the quantum convolutional neural
    network (LPPC).
    """
    ## DEVICE
    # *1* Define number of wires for device (easier to switch between device types):
    device_wires = 6
    # *2* Select device for class:
    # device = qml.device('default.qubit.jax', wires=device_wires)
    device = qml.device("default.qubit", wires=device_wires)
    # device = qml.device("default.mixed", wires=device_wires)

    def __init__(self):
        ## CLASSES:
        self.qmo = qmath_ops
        ## QUBITS:
        self.n_qubits = 6 # Set 'n_qubits' equal to desired qubit configuration (for us, 6)
        self.n_qubits_draw = 2 # 2 qubit config for DRAWQC
        ## ACTIVE QUBITS:
        self.active_qubits = 6 # Set 'active_qubits' equal to 'n_qubits''
        ## WIRES:
        self.wires = 6
        self.n_wires = 6
        self.num_wires = 6
        self.n_wires_draw = 2 # (For drawings)
    
    # ----------------------------------------------------
    #     ORIGINAL CIRCUIT AND LAYER FUNCTIONS (LPPC)
    # ----------------------------------------------------

    # ******* ORIGINAL CONVOLUTIONAL LAYER *******:
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
        pool_operators = self.qmo.generate_gell_mann(self.qmo, (2**n_qubits))

        # Loop over all active qubits (in pairs):
        index = 0
        while index + 2 < len(active_qubits):
            q1 = active_qubits[index]   # First qubit
            q2 = active_qubits[index+1] # Second qubit

            # Get Convolutional Operator 'v_conv' from 'get_conv_op' and weights:
            v_conv = self.qmo.get_conv_op(self.qmo,
                                               pool_operators, weights[index])  # V -> Conv. Operator
            
            # Apply Controlled Unitary Operation with convolutional operator:
            qml.ControlledQubitUnitary(v_conv, control_wires=[q1], wires=[q2])

            index += 2

    # ******* ORIGINAL POOLING LAYER *******:
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
        pool_operators = self.qmo.generate_gell_mann(self.qmo, 2**len(n_qubits))

        # Loop over all active qubits (in pairs):
        index = 0
        while index + 2 < len(active_qubits):
            q1 = active_qubits[index]   # First qubit
            q2 = active_qubits[index+1] # Second qubit

            # Get convolutional operators V1 and V2 from pool operators and weights:
            # V1 -> First set of weights
            v1 = self.qmo.get_conv_op(self.qmo,
                                           pool_operators, weights[index])
            # V2 -> Second set of weights
            v2 = self.qmo.get_conv_op(self.qmo,
                                           pool_operators, weights[index+1])
            
            # Get convolutional operators "V1_pool" and "v2_pool" from pool operators and weights:
            v1_pool = self.qmo.controlled_pool(self.qmo, v1)
            v2_pool = self.qmo.controlled_pool(self.qmo, v2)

            # Apply Hadamard Gate to first qubit:
            qml.Hadamard(wires=q1)

            # Apply FIRST Controlled Unitary Operation:
            qml.ControlledQubitUnitary(v1_pool, control_wires=[q1], wires=[q2])

            # Apply SECOND Controlled Unitary Operation:
            qml.ControlledQubitUnitary(v2_pool, control_wires=[q1], wires=[q2])

            index += 2

        # Update active qubits by pooling 1 / 2 qubits:
        self.active_qubits = active_qubits[::2]

    # ******* ORIGINAL FULLY CONNECTED LAYER *******:
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

    # ******* ORIGINAL FULLY CONNECTED LAYER (WITH TWO-QUBIT UNITARIES) *******:
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
        fc_mats = self.qmo.generate_gell_mann(self.qmo, (2**len(n_qubits)))
        fc_op = self.qmo.get_conv_op(self.qmo, fc_mats, params)

        # Apply Fully Connected Operator to active qubits:
        qml.QubitUnitary(fc_op, wires=active_qubits)

    # ******* ORIGINAL QUANTUM CIRCUIT *******:
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


# ============================================================
#            ORIGINAL CIRCUIT DRAWING CLASS (LPPC)
# ============================================================
    

class DrawLPPC(LayersLPPC):
    """
    Contains functions for drawing different layer functions relevant to the
    quantum convolutional neural network, including convolutional layers, pooling layers,
    fully connected layers, and quantum circuits. Defaults to a 2-Qubit representation of
    each layer, as well as defaulting to the most recent version of each layer function if
    none is specified (LPPC).
    """
    def __init__(self):
        ## CLASSES:
        self.lppc_layers = LayersLPPC()
        # QUBITS:
        self.n_qubits = 6 # Set 'n_qubits' equal to desired qubit configuration
        self.n_qubits_draw = 2 # 2 qubit config for drawings
        # ACTIVE QUBITS:
        self.active_qubits = 6 # Set 'active_qubits' equal to 'n_qubits'
        # WIRES:
        self.wires = 6
        self.n_wires = 6
        self.num_wires = 2 # For drawings
    
    # ----------------------------------------------------
    #         ORIGINAL DRAWING FUNCTIONS (LPPC)
    # ----------------------------------------------------

    # ******* ORIGINAL DRAW POOLING LAYER CIRCUIT *******:
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

    # ******* ORIGINAL DRAW CONVOLUTIONAL LAYER CIRCUIT *******:
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

    # ******* ORIGINAL DRAW FULLY CONNECTED LAYER CIRCUIT *******:
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


# ============================================================
#            NEW OPTIMIZATION AND TRAINING CLASS
# ============================================================


class TrainQC(LayersQC):
    """
    Contains functions for optimization steps in a Quantum Convolutional Neural Network (QCNN).
    It depends on the layer functions provided in the QCircuitLPPC class.
    """
    # qcnn_layers = LayersLPPC()
    def __init__(self):
        ## CLASSES:
        self.qc_layers = LayersQC()  # Instantiate LayersQC (once)
        # self.qcnn_layers = LayersQC()
        # QUBITS:
        self.n_qubits = 6 # Set 'n_qubits' equal to desired qubit configuration (for us, 6)
        self.n_qubits_draw = 2 # 2 qubit config for DRAWQC
        # ACTIVE QUBITS:
        self.active_qubits = 6 # Set 'active_qubits' equal to 'n_qubits''
        # WIRES:
        self.wires = 6
        self.n_wires = 6
        self.num_wires = 2 # For drawings

        # TRAINING:
        self.num_train = 2
        self.num_test = 2

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

    # ----------------------------------------------------
    #             NEW OPTIMIZATION FUNCTIONS
    # ----------------------------------------------------

    # ******* COMPUTE (LABEL) OUTPUT *******:
    @staticmethod
    @jax.jit # (JAX NumPy use)
    def compute_out(weights, weights_last, features, labels):
        """
        Computes the output of the corresponding label in the QCNN model.
        """
        # Instantiate 'LayersQC' class instance:
        qc_layers = LayersQC()

        cost = lambda weights, weights_last, feature, label: qc_layers.conv_net(qc_layers, weights, weights_last, feature)[
            label
        ]

        return jax.vmap(cost, in_axes=(None, None, 0, 0), out_axes=0)(
            weights, weights_last, features, labels
        )

    # ******* NEW COMPUTE ACCURACY *******:
    # @jax.jit(static_argnums=(0, 1, 2, 3))
    # @jax.jit
    # @jax.jit(static_argnames=['weights', 'weights_last', 'features', 'labels'])
    @staticmethod
    def compute_accuracy(weights, weights_last, features, labels):
        """
        Computes the accuracy over the provided features and labels.
        """
        out = TrainQC.compute_out(weights, weights_last, features, labels)

        return jnp.sum(out > 0.5) / len(out)

    # ******* NEW COMPUTE COST *******:
    @staticmethod
    def compute_cost(weights, weights_last, features, labels):
        """
        Computes the cost over the provided features and labels.
        """
        # compute_cost_lambda = lambda w, wl, f, l: self.compute_cost(w, wl, f, l) # lambda cost function
        # value_and_grad = jax.jit(jax.value_and_grad(compute_cost_lambda, argnums=[0, 1]))
        out = TrainQC.compute_out(weights, weights_last, features, labels)

        return 1.0 - jnp.sum(out) / len(labels)

    # ******* INITIALIZE WEIGHTS *******:
    @staticmethod
    @jax.jit # (JAX NumPy use)
    def init_weights():
        """
        Initializes random weights for the QCNN model.
        """
        weights = pnp.random.normal(loc=0, scale=1, size=(81, 1), requires_grad=True)
        weights_last = pnp.random.normal(loc=0, scale=1, size=4 ** 3 - 1, requires_grad=True) 
        # weights = pnp.random.normal(loc=0, scale=1, size=(81, 2), requires_grad=True) # (Possible NP -> JNP)
        # weights_last = pnp.random.normal(loc=0, scale=1, size=4 ** 2 - 1, requires_grad=True) # (Possible NP -> JNP)

        return jnp.array(weights), jnp.array(weights_last)
    
    # ******* NEW TRAIN QCNN *******:
    @staticmethod
    def train_qcnn(n_train, n_test, n_epochs, use_moments):
        """
        Trains data for the QCNN model.
        """

        ## DATA PREPARATION:
        # ------------------------------------------------------------------------------------------
        if use_moments is True:
            # bin_type = "100GeV-1TeV"
            bin_type = "1TeV-10TeV"
            # bin_type = "10TeV-100TeV"
            # bin_type = "100TeV-1PeV"
            features, labels = LoadPhotonData.prepare_moments_data_jax(energy_bin=bin_type)
        else: 
            # Prepare data (NumPy):
            features_np, labels_np = LoadDataQC.prepare_data()
            features = jnp.array(features_np) # NP -> JAX Arrays
            labels = jnp.array(labels_np) # NP -> JAX Arrays
        # ------------------------------------------------------------------------------------------

        ## JAX.JIT CONFIGURATION:
        # ------------------------------------------------------------------------------------------
        use_wrapped_version_train = False  # TRUE -> jax.jit-wrapped version, FALSE -> direct call

        if use_moments is True:
            # Use LoadPhotonData.load_moments_data_jax
            x_train, y_train, x_test, y_test = LoadPhotonData.load_moments_data_jax(
                n_train=n_train, n_test=n_test, features=features, labels=labels
            )
        else:
            # TRUE (JAX.JIT-WRAPPED):
            if use_wrapped_version_train is True:
                # Load data using wrapped function:
                load_digits_data_jax_wrapped = jax.jit(LoadDataQC.load_digits_data_jax)
                x_train, y_train, x_test, y_test = load_digits_data_jax_wrapped(
                    n_train, n_test, features, labels
                )
            # FALSE (DIRECT CALL):
            else:
                x_train, y_train, x_test, y_test = LoadDataQC.load_digits_data_jax(
                    n_train=n_train, n_test=n_test, features=features, labels=labels
                )
        # ------------------------------------------------------------------------------------------

        # Define lambda cost function
        compute_cost_lambda = lambda w, wl, f, l: TrainQC.compute_cost(w, wl, f, l)

        # Update 'value_and_grad', weights with lambda cost:
        value_and_grad = jax.jit(jax.value_and_grad(compute_cost_lambda, argnums=[0, 1]))

        weights, weights_last = TrainQC.init_weights()

        cosine_decay_scheduler = optax.cosine_decay_schedule(0.1, decay_steps=n_epochs, alpha=0.95)
        optimizer = optax.adam(learning_rate=cosine_decay_scheduler)
        opt_state = optimizer.init((weights, weights_last))

        train_cost_epochs, test_cost_epochs, train_acc_epochs, test_acc_epochs = [], [], [], []

        for step in range(n_epochs):
            train_cost, grad_circuit = value_and_grad(weights, weights_last, x_train, y_train)
            updates, opt_state = optimizer.update(grad_circuit, opt_state)
            weights, weights_last = optax.apply_updates((weights, weights_last), updates)

            train_cost_epochs.append(train_cost)

            train_acc = TrainQC.compute_accuracy(weights, weights_last, x_train, y_train)
            train_acc_epochs.append(train_acc)

            test_out = TrainQC.compute_out(weights, weights_last, x_test, y_test)
            test_acc = jnp.sum(test_out > 0.5) / len(test_out)
            test_acc_epochs.append(test_acc)
            test_cost = 1.0 - jnp.sum(test_out) / len(test_out)
            test_cost_epochs.append(test_cost)

        # Create JAX array for 'n_train':
        n_train_list = [n_train] * n_epochs # ORIGINAL ('n_train')
            
        return dict(
            n_train=n_train_list,
            step=jnp.arange(1, n_epochs + 1, dtype=int), # NP -> JNP
            train_cost=train_cost_epochs,
            train_acc=train_acc_epochs,
            test_cost=test_cost_epochs,
            test_acc=test_acc_epochs,
        )

    # @jax.jit # (JIT-compiled use)
    # ******* RUN QCNN TRAINING ITERATIONS *******:
    @staticmethod
    def run_iterations(n_train, n_test, n_epochs, n_reps, use_moments):
        """
        Runs selected number of iterations of training loop for the QCNN model.
        """

        # Original (below): ["train_acc", "train_cost", "test_acc", "test_cost", "step", "n_train"]
        # Current: ["n_train", "step", "train_cost", "train_acc", "test_cost", "test_acc"]
        results_df = pd.DataFrame(
            columns=["n_train", "step", "train_cost", "train_acc", "test_cost", "test_acc"]
        )

        for _ in range(n_reps):
            results = TrainQC.train_qcnn(n_train, n_test, n_epochs, use_moments)
            results_df = pd.concat(
                [results_df, pd.DataFrame.from_dict(results)], axis=0, ignore_index=True
            )

        return results_df # (commented out?)
    
    # @jax.jit # (JIT-compiled use)
    # ******* COMPUTE AGGREGATED TRAINING RESULTS *******:
    @staticmethod
    def compute_aggregated_results(n_train, n_test, n_epochs, n_reps, use_moments=False):
        """
        Function to run training iterations for multiple sizes and aggregate the results.
        """
        # Run training for multiple sizes:
        # train_sizes = [2, 5, 10, 20, 40, 80]
        # Single-sized training:
        # train_sizes = [2] # for MNIST data
        train_sizes = [80] # for MOI data
        results_df = TrainQC.run_iterations(n_train, n_test, n_epochs, n_reps, use_moments)
        for n_train in train_sizes[1:]:
            results_df = pd.concat([results_df, TrainQC.run_iterations(n_train, n_test,
                                                        n_epochs, n_reps, use_moments)])
        
        return results_df
    
    # ******* PLOT AGGREGATED TRAINING RESULTS *******:
    @staticmethod
    def plot_aggregated_results(results_df, n_train, steps=100, 
                            title_loss='Train and Test Losses', 
                            title_accuracy='Train and Test Accuracies',
                            markevery=10,
                            save_fig=False):
        """
        Function to aggregate and plot results from a DataFrame.
        
        Args:
        esults_df (DataFrame): Contains the results to aggregate and plot
        steps (int): Specific steps or epochs to consider for the final loss difference calculation
        title_loss (string): Title for the loss plot (Optional)
        title_accuracy (string): Title for the accuracy plot (Optional)
        markevery (int): Interval at which markers are displayed (Optional)
        """
        # train_sizes_1 = [2]
        # train_sizes_2 = [2, 5]
        # train_sizes_3 = [2, 5, 10]

        # aggregate dataframe
        df_agg = results_df.groupby(["n_train", "step"]).agg(["mean", "std"])
        df_agg = df_agg.reset_index()

        sns.set_style('whitegrid')
        # colors = sns.color_palette()
        # colors = [sns.color_palette()[0]] # for n_train = 2
        # colors = [sns.color_palette()[1]] # for n_train = 5
        colors = [sns.color_palette()[2]] # for n_train = 10
        fig, axes = plt.subplots(ncols=2, figsize=(16.5, 5))

        generalization_errors = []
        # train_sizes = [2] # original
        train_sizes = [n_train] # new
        # plot losses and accuracies
        for i, n_train in enumerate(train_sizes):
            df = df_agg[df_agg.n_train == n_train]

            dfs = [df.train_cost["mean"], df.test_cost["mean"], df.train_acc["mean"], df.test_acc["mean"]]
            lines = ["o-", "x--", "o-", "x--"]
            labels = [fr"$N={n_train}$", None, fr"$N={n_train}$", None]
            axs = [0, 0, 1, 1]

            for k in range(4):
                ax = axes[axs[k]]
                ax.plot(df.step, dfs[k], lines[k], label=labels[k], markevery=markevery, color=colors[i], alpha=0.8)


            # plot final loss difference
            dif = df[df.step == steps].test_cost["mean"] - df[df.step == steps].train_cost["mean"]
            generalization_errors.append(dif)

        # format loss plot
        ax = axes[0]
        ax.set_title(title_loss, fontsize=14)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')

        # format generalization error plot
        # ax = axes[1]
        # ax.plot(train_sizes, generalization_errors, "o-", label=r"$gen(\alpha)$")
        # ax.set_xscale('log')
        # ax.set_xticks(train_sizes)
        # ax.set_xticklabels(train_sizes)
        # ax.set_title(r'Generalization Error $gen(\alpha) = R(\alpha) - \hat{R}_N(\alpha)$', fontsize=14)
        # ax.set_xlabel('Training Set Size')

        # format loss plot
        ax = axes[1]
        ax.set_title(title_accuracy, fontsize=14)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0.5, 1.05)

        legend_elements = [
            mpl.lines.Line2D([0], [0], label=f'N={n}', color=colors[i]) for i, n in enumerate(train_sizes)
            ] + [
            mpl.lines.Line2D([0], [0], marker='o', ls='-', label='Train', color='Black'),
            mpl.lines.Line2D([0], [0], marker='x', ls='--', label='Test', color='Black')
            ]

        axes[0].legend(handles=legend_elements, ncol=3)
        axes[1].legend(handles=legend_elements, ncol=3)
        #axes[1].set_yscale('log', base=2)

        if save_fig is True:
            current_date = datetime.now().strftime('%m-%d-%Y') # current date in format MM-DD-YYYY
            save_path_results = f'qcnn-figures/qcnn_ntrain{n_train}_results_{current_date}.png' # save path
            plt.savefig(save_path_results, dpi=400, bbox_inches="tight") # high-res image
        plt.show()


# ============================================================
#       ORIGINAL OPTIMIZATION AND TRAINING CLASS (LPPC)
# ============================================================


class TrainLPPC(DrawLPPC):
    """
    Contains functions for optimization steps in a Quantum Convolutional Neural Network (QCNN).
    It depends on the layer functions provided in the QCircuitLPPC class.
    """
    def __init__(self):
        self.lppc_layers = LayersLPPC()
        self.lppc_draw = DrawLPPC()
        # QUBITS:
        self.n_qubits = 6 # Set 'n_qubits' equal to desired qubit configuration (for us, 6)
        self.n_qubits_draw = 2 # 2 qubit config for DRAWQC
        # ACTIVE QUBITS:
        self.active_qubits = 6 # Set 'active_qubits' equal to 'n_qubits''
        # WIRES:
        self.wires = 6
        self.n_wires = 6
        self.num_wires = 2 # For drawings

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
    
    # ----------------------------------------------------
    #       ORIGINAL OPTIMIZATION FUNCTIONS (LPPC)
    # ----------------------------------------------------
    
    # ******* SELECT QCNN OPTIMIZER *******:
    def qcnn_opt_select(opt_methods=None, opt_num=None):
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

    # ******* ORIGINAL MEAN SQUARED ERROR COST *******:
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

    # ******* ORIGINAL STOCHASTIC GRADIENT DESCENT *******:
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

    # ******* ORIGINAL ACCURACY *******:
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


# **************************************************************************************************
#                                                MAIN
# **************************************************************************************************


def main():
    return


if __name__ == "__main__":
    main()
