########################################## OPERATORS.PY ############################################

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


# ==============================================================================================
#                         QUANTUM AND MATHEMATICAL OPERATIONS CLASS
# ==============================================================================================


class QuantumMathOps:
    """
    Class that contains custom quantum computing operators, relevant mathematical constructs,
    qubit functions, and other related operations.
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
    #         ORIGINAL QUANTUM OPERATOR FUNCTIONS (LPPC)
    # -----------------------------------------------------------

    # ******* Original General Rotation Gate *******:
    def GRot_orig(self, params, wire):
        """
        General Rotation Gate to a given Qubit (original version).
        """
        qml.Rot(params[0], params[1], params[2], wire=wire)
        # return qml.expval(qml.PauliZ(wire))

    # ******* Original Typecasting Weights Function *******:
    def typecast_weights_orig(self, params):
        """
        Transforms the parameters to a Torch tensor of type 'complex128' with 
        'requires_grad' set to 'True' (original version).
        """
        params_complex = params.astype(np.complex128) # convert to type 'complex128'
        params_alpha = torch.tensor(params_complex, requires_grad=True) # convert to Torch tensor
        params = np.array(params_alpha) # Convert to Numpy array

        return params

    # ******* Original Broadcasting Weights Function *******:
    def broadcast_weights_orig(self, params, n_qubits=None, check_qubits=True):
        """
        Transforms the weights into the appropriate broadcasting form for the given
        number of qubits (original version).
        """
        # Check Qubit Configuration:
        if check_qubits is True:
            # Number of Qubits:
            if n_qubits is None:
                n_qubits = self.n_qubits
            
        params_flat = params.reshape(-1)
        # params = np.array(params, requires_grad=True)
        params = np.array(params_flat, requires_grad=True) # Convert to Numpy array

        return params

    # ******* Original Preparing Weights Function *******:
    def prep_weights_orig(self, params, n_qubits=None, check_qubits=True, dtype_key=None):
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
        # *1* Get dtype Selection:
        np_dtype, torch_dtype = dtype_dict.get(dtype_key)

        # Convert to Specified dtype:
        if np_dtype is not None:
            params_alpha = params_flat.astype(np_dtype)
        else:
            params_alpha = params_flat.astype(dtype_dict['default'][0])

        # *2* Convert to Torch tensor:
        if torch_dtype is not None:
            params = torch.tensor(params_alpha, dtype=torch_dtype,
                                  requires_grad=True)
        else:
            params = torch.tensor(params_alpha, dtype=dtype_dict['default'][1])
        #-----------------------------------------------------------------------

        return params

    # -----------------------------------------------------------
    #         NEW AND ESSENTIAL QUANTUM OPERATOR FUNCTIONS
    # -----------------------------------------------------------

    # ******* Basis Matrix *******:
    def b_mat(self, i, j, n):
        """
        Generates an n x n matrix of 0s with the i,j th entry is a one.
        This is the i,j th basis vector on the space of n x n real matrices.
        Returns np.array of floats, shape (n,n).

        List of Parameters:
        -> param 'i': int, row index (must be < n)
        -> param 'j': int, column index (must be < n)
        -> param 'n': int, dimension of the matrices
        """
        basis_matrix = np.zeros((n, n), dtype=np.float64) # ORIGINAL: dtype=np.float32
        basis_matrix[i, j] = 1.0
        return basis_matrix

    # ******* Gell Mann Matrices *******:
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
                diag_mat = coeff * (sum_diag - n * (self.b_mat(self, k + 1, k + 1, order)))
                gm_matrices.append(diag_mat)

        return gm_matrices

    # ******* Convolutional Operator *******:
    def get_conv_op(self, mats, params):
        """
        Parametrizes the convolutional operator according to Gell-Mann matrices scaled by 
        trainable parameters, this method generates the relevant applicable operator.
        """
        final = np.zeros(mats[0].shape, dtype=np.complex128)
        for mat, param in zip(mats, params):
            final += param * mat
        return expm(complex(0, -1) * final)

    # ******* Controlled Pool Operator *******:
    def controlled_pool(self, mat):
        """
        Generates the matrix corresponding the controlled - mat operator. Inputs Numpy array,
        shape (2x2) for the controlled operator and returns the final controlled-mat operator.
        """
        i_hat = np.array([[1.0, 0.0], [0.0, 0.0]])
        j_hat = np.array([[0.0, 0.0], [0.0, 1.0]])
        identity = i_hat + j_hat
        return np.kron(i_hat, identity) + np.kron(j_hat, mat)

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
