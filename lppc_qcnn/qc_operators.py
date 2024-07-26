############################################ QC_OPERATORS.PY ############################################

### ***** IMPORTS / DEPENDENCIES *****:

## PENNYLANE:
import pennylane as qml
from pennylane import numpy as np

## JAX:
import jax;
## JAX CONFIGURATIONS:
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
# import jax.experimental.sparse as jsp # (NOT ACCESSED)
import jax.scipy.linalg as jsl # (NOT ACCESSED)

# -------------------------------------------------
## TORCHVISION (FOR OPERATORS):
import torch 
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader

## TENSORFLOW (FOR OPERATORS):
# import tensorflow as tf
# from tensorflow.keras.datasets import mnist
# -------------------------------------------------

## RNG:
seed = 0
rng = np.random.default_rng(seed=seed) # ORIGINAL
rng_jax = jax.random.PRNGKey(seed=seed) # *1* using JAX
rng_jax_arr = jnp.array(jax.random.PRNGKey(seed=seed)) # *2* using JAX

## OTHER:
# from glob import glob
# from scipy.linalg import expm # (NOT ACCESSED)

### ***** PACKAGE(S) *****:
# ************************************************************************************
# OPERATORS.PY (SELF):
# from .qc_operators import QuantumMathOps as qmath_ops # QuantumMathOps()
# from .qc_operators import PenguinsQMO as lppc_qmo # PenguinsQMO() # (NOT ACCESSED)
# Example Usage(s) (Instance Method):
# -> QuantumMathOps():
#       qmo_obj = qmath_ops
#       qmo_obj.sample_function(*args, **kwargs)
# -> PeguinsQMO():
#       lppc_qmo_obj = lppc_qmo
#       lppc_qmo_obj.sample_function(*args, **kwargs)
# ************************************************************************************


# ============================================================
#              QUANTUM AND MATH OPERATIONS CLASS
# ============================================================


class QuantumMathOps():
    """
    Class that contains custom quantum computing operators, relevant mathematical constructs,
    qubit functions, and other related operations.
    """
    def __init__(self):
        # QUBITS:
        self.n_qubits = 6 # Set 'n_qubits' equal to desired qubit configuration (for us, 6)
        self.n_qubits_draw = 2 # 2 qubit config for DRAWQC
        # ACTIVE QUBITS:
        self.active_qubits = 6 # Set 'active_qubits' equal to 'n_qubits'
        # WIRES:
        self.wires = 6
        self.n_wires = 6
        self.num_wires = 2 # For drawings
    
    # ----------------------------------------------------
    #        QUANTUM OPERATOR FUNCTIONS (ESSENTIAL)
    # ----------------------------------------------------

    # ******* BASIS MATRIX *******:
    def b_mat(self, i, j, n):
        """
        Generates an n x n matrix of 0s with the i,j th entry is a one.
        This is the i,j th basis vector on the space of n x n real matrices.
        Returns np.array of floats, shape (n,n).

        Args:
        -> param 'i': int, row index (must be < n)
        -> param 'j': int, column index (must be < n)
        -> param 'n': int, dimension of the matrices
        """
        basis_matrix = np.zeros((n, n), dtype=np.float32)
        basis_matrix[i, j] = 1.0

        return basis_matrix

    # ******* GELL MANN MATRICES *******:
    def generate_gell_mann(self, order):
        """
        Generates a list of np.arrays which represent Gell Mann matrices of order 'order'.
        Example: 'order' = 2
         -> gm_matrices = [ [[0,  1],
                                    [1,  0]] ,

                                    [[0, -i]
                                    [i,  0]] ,

                                    [[1,  0],
                                    [0, -1]] ]
        """
        gm_matrices = []

        for k in range(order):
            j = 0
            while j < k:
                sym = self.b_mat(self, j, k, order) + self.b_mat(self, k, j, order)
                anti_sym = complex(0.0, -1.0) * (self.b_mat(self, j, k, order) - self.b_mat(self, k, j, order))
                gm_matrices.append(sym)
                gm_matrices.append(anti_sym)
                j += 1

            if k < (order - 1):
                n = k + 1
                coeff = np.sqrt(2 / (n * (n + 1)))
                sum_diag = self.b_mat(self, 0, 0, order)
                for i in range(1, k + 1):
                    sum_diag += self.b_mat(self, i, i, order)
                diag_mat = coeff * (sum_diag - n * (self.b_mat(self, (k + 1), (k + 1), order)))
                gm_matrices.append(diag_mat)

        return gm_matrices

    # ******* CONVOLUTIONAL OPERATOR *******:
    def get_conv_op(self, mats, params):
        """
        Parametrizes the convolutional operator according to Gell-Mann matrices scaled by 
        trainable parameters, this method generates the relevant applicable operator.
        """
        final = np.zeros(mats[0].shape, dtype=np.complex128)
        for mat, param in zip(mats, params):
            final += param * mat
        
        # return jsl.expm(complex(0, -1) * final) (non-JAX)
        return jsl.expm(complex(0, -1) * final)

    # ******* CONTROLLED POOL OPERATOR (WITH NUMPY) *******:
    def controlled_pool_numpy(self, mat):
        """
        Generates the matrix corresponding the controlled - mat operator using NumPy. Inputs 
        Numpy array, shape (2x2) for the controlled operator and returns the final 
        controlled-mat operator (LPPC).
        """
        i_hat = np.array([[1.0, 0.0], [0.0, 0.0]])
        j_hat = np.array([[0.0, 0.0], [0.0, 1.0]])
        identity = i_hat + j_hat

        return np.kron(i_hat, identity) + np.kron(j_hat, mat)

    # ----------------------------------------------------
    #          QUANTUM OPERATOR FUNCTIONS (NEW)
    # ----------------------------------------------------
    
    # ******* NEW CONTROLLED POOL OPERATOR *******:
    def controlled_pool(self, mat):
        """
        Generates the matrix corresponding to the controlled-mat operator. Inputs JAX array,
        shape (2x2) for the controlled operator and returns the final controlled-mat operator.
        """
        i_hat = jnp.array([[1.0, 0.0], [0.0, 0.0]])
        j_hat = jnp.array([[0.0, 0.0], [0.0, 1.0]])
        identity = i_hat + j_hat

        return jnp.kron(i_hat, identity) + jnp.kron(j_hat, mat)
    
    # ******* UNIFORMLY CONTROLLED ROTATION *******:
    def generate_uniformly_controlled_rotation(self, params, control_qubit_indicies,
                                               target_qubit_index, axis='z'):
        """
        Applies a uniformly controlled rotation to a target qubit based on control qubits.

        Args:
            -> params (np.array): Array with the rotation angles for the uniformly controlled rotations.
            -> control_qubit_indicies (list[int]): List of indices of the control qubits.
            -> target_qubit_index (int): Index of the target qubit.
            -> axis (str): The axis of rotation ('x', 'y', or 'z'). Default is 'z'.
        """
        num_control_qubits = len(control_qubit_indicies)

        divisors = range(num_control_qubits - 1, -1, -1)   # Starts from largest divisor to smallest
        divisors = [2**i for i in divisors]

        for iteration_num, theta in zip(range(1, 2**num_control_qubits + 1), params):
            if axis == 'z':
                qml.RZ(theta, target_qubit_index)
            elif axis == 'y':
                qml.RY(theta, target_qubit_index)
            else:
                qml.RX(theta, target_qubit_index)

            for divisor in divisors:
                if iteration_num % divisor == 0:
                    control_element = int((num_control_qubits - 1) - np.log2(divisor))
                    qml.CNOT(control_qubit_indicies[control_element], target_qubit_index)
                    break


# ============================================================
#       ORIGINAL QUANTUM OPERATOR FUNCTIONS CLASS (LPPC)      
# ============================================================


class PenguinsQMO():
    """
    Class that contains custom quantum computing operators, relevant mathematical constructs,
    qubit functions, and other related operations (LPPC).
    """
    def __init__(self):
        self.qmo = QuantumMathOps()
        # QUBITS:
        self.n_qubits = 6 # Set 'n_qubits' equal to desired qubit configuration (for us, 6)
        self.n_qubits_draw = 2 # 2 qubit config for DRAWQC
        # ACTIVE QUBITS:
        self.active_qubits = 6 # Set 'active_qubits' equal to 'n_qubits'
        # WIRES:
        self.wires = 6
        self.n_wires = 6
        self.num_wires = 2 # For drawings
    
    # ----------------------------------------------------
    #     ORIGINAL QUANTUM OPERATOR FUNCTIONS (LPPC)
    # ----------------------------------------------------

    # ******* GENERAL ROTATION GATE (LPPC) *******:
    def GRot_lppc(params, wire):
        """
        General Rotation Gate to a given Qubit (original version).
        """
        qml.Rot(params[0], params[1], params[2], wire=wire)
        # return qml.expval(qml.PauliZ(wire))

    # ******* TYPECASTING WEIGHTS (LPPC) *******:
    def typecast_weights_lppc(params):
        """
        Transforms the parameters to a Torch tensor of type 'complex128' with 
        'requires_grad' set to 'True' (original version).
        """
        params_complex = params.astype(np.complex128) # convert to type 'complex128'
        params_alpha = torch.tensor(params_complex, requires_grad=True) # convert to Torch tensor
        params = np.array(params_alpha) # Convert to Numpy array

        return params

    # ******* BROADCASTING WEIGHTS (LPPC) *******:
    def broadcast_weights_lppc(params):
        """
        Transforms the weights into the appropriate broadcasting form for the given
        number of qubits (original version).
        """
            
        params_flat = params.reshape(-1)
        # params = np.array(params, requires_grad=True)
        params = np.array(params_flat, requires_grad=True) # Convert to Numpy array

        return params

    # ******* TYPECASTING WEIGHTS (LPPC) *******:
    def prep_weights_lppc(self, params, n_qubits=None, check_qubits=True, dtype_key=None):
        """
        Prepares weights for passing into QCNN: Transforms the weights into the appropriate
        broadcasting form for the given number of qubits (size of 2^{n_qubits}, pad with zeros).
        Then transforms the parameters to a Torch tensor with the specified dtype and 
        'requires_grad' set to 'True' (original version).
        
        List of Available Datatypes (THREE total):
        -> 'complex' : Converts to 'complex128'
        -> 'float' : Converts to 'float64'
        -> 'default' : No conversion, keeps original type
        """
        # Check Qubit Configuration:
        if check_qubits is True:
            # Number of Qubits:
            if n_qubits is None:
                n_qubits = self.n_qubits

        # Dictionary for dtype selection
        dtype_dict = {
            'complex': (np.complex128, torch.complex128),
            'float': (np.float64, torch.float64),
            'default': (np.int64, torch.long)
        }

        params_flat = params.reshape(-1)
        opt_length = (2**n_qubits)

        # Pad Parameters with Zeros (As Needed):
        if len(params_flat) < opt_length:
            params_flat = np.pad(params_flat, (0, opt_length - len(params_flat)),
                                mode='constant', constant_values=0.0)
        
        # SHAPE AND DTYPE CONVERSION:
        #-----------------------------------------------------------------------
        # Get dtype selection:
        np_dtype, torch_dtype = dtype_dict.get(dtype_key)

        # Convert to specified dtype:
        if np_dtype is not None:
            params_alpha = params_flat.astype(np_dtype)
        else:
            params_alpha = params_flat.astype(dtype_dict['default'][0])

        # Convert to torch tensor:
        if torch_dtype is not None:
            params = torch.tensor(params_alpha, dtype=torch_dtype,
                                requires_grad=True)
        else:
            params = torch.tensor(params_alpha, dtype=dtype_dict['default'][1])
        #-----------------------------------------------------------------------

        return params


# **************************************************************************************************
#                                                MAIN
# **************************************************************************************************


def main():
    return


if __name__ == "__main__":
    main()
