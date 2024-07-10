########################################## gellmann_ops.py ############################################

### IMPORTS / DEPENDENCIES:

# PennyLane:
import pennylane as qml
from pennylane import numpy as np

# TorchVision:
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Other:
import scipy
from scipy.linalg import expm

# Plotting:
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update(mpl.rcParamsDefault)


################### GELL MANN MATRIX OPERATION CLASS ######################


class GellMannOps:
    """
    Class for generating Gell-Mann matrices and related operations.
    """
    def __init__(self):
        self.n_qubits_mnist = 10
        self.n_qubits_lppc = 4
        self.n_qubits_test = 2
        self.num_active_qubits = 10
        self.num_qubits = 10
        self.active_qubits = self.num_active_qubits
        self.n_qubits = self.n_qubits_mnist
        self.num_wires = 2 # For QCNN Drawings


    # BASIS MATRIX:
    @staticmethod
    def b_mat(i, j, n):
        """
        Generates an n x n matrix of 0s with the i,j th entry is a one.
        This is the i,j th basis vector on the space of n x n real matrices

        :param i: int, row index (must be < n)
        :param j: int, column index (must be < n)
        :param n: int, dimension of the matrices
        :return: np.array of floats, shape (n,n)
        """
        basis_matrix = np.zeros((n, n), dtype=np.float32)
        basis_matrix[i, j] = 1.0
        return basis_matrix


    # GELL-MANN MATRICES:
    def generate_gell_mann(self, order):
        """
        Generates a list of np.arrays which represent Gell Mann matrices of order 'order'.
        eg: order = 2
        gm_matrices = [ [[0,  1],
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
                sym = self.b_mat(j, k, order) + self.b_mat(k, j, order)
                anti_sym = complex(0.0, -1.0) * (self.b_mat(j, k, order) - self.b_mat(k, j, order))
                gm_matrices.append(sym)
                gm_matrices.append(anti_sym)
                j += 1

            if k < (order - 1):
                n = k + 1
                coeff = np.sqrt(2 / (n * (n + 1)))
                sum_diag = self.b_mat(0, 0, order)
                for i in range(1, k + 1):
                    sum_diag += self.b_mat(i, i, order)
                diag_mat = coeff * (sum_diag - n * (self.b_mat(k + 1, k + 1, order)))
                gm_matrices.append(diag_mat)

        return gm_matrices


    # CONVOLUTIONAL OPERATOR:
    @staticmethod
    def get_conv_op(mats, params):
        """
        Parametrizes the convolutional operator according to Gell-Mann matrices scaled by trainable parameters,
        this method generates the relevant applicable operator.
        """
        final = np.zeros(mats[0].shape, dtype=np.complex128)
        for mat, param in zip(mats, params):
            final += param * mat
        return expm(complex(0, -1) * final)


    # CONTROLLED POOL OPERATOR:
    @staticmethod
    def controlled_pool(mat):
        """
        Generates the matrix corresponding the controlled - mat operator.

        :param mat: np.array, shape (2x2) for the controlled operator
        :return: np.array, the final controlled-mat operator
        """
        i_hat = np.array([[1.0, 0.0], [0.0, 0.0]])
        j_hat = np.array([[0.0, 0.0], [0.0, 1.0]])
        identity = i_hat + j_hat
        return np.kron(i_hat, identity) + np.kron(j_hat, mat)


    # CUSTOM ROTATION GATE:
    @staticmethod
    def G_Rot(weights, wire):
        """General Rotation Gate to Qubit."""
        qml.Rot(weights[0], weights[1], weights[2], wires=wire)


################### PARAMETER OPERATIONS HELPER CLASS ######################
        
        
class ParamOps(GellMannOps):
    """
    Class for handling parameter transformations and broadcasting, dependent on GellMannOps.
    """
    def __init__(self):
        super().__init__()

    # TYPECASTING WEIGHTS:
    @staticmethod
    def transform_params(params):
        """
        Transforms the parameters to a Torch tensor of type 'complex128' with 
        'requires_grad' set to 'True'.
        """
        params_complex = params.astype(np.complex128) # convert to type 'complex128'
        params_alpha = torch.tensor(params_complex, requires_grad=True) # convert to Torch tensor
        params = np.array(params_alpha) # Convert to Numpy array

        return params
    

    # BROADCASTING (VERSION #1):
    @staticmethod
    def broadcast_params_V1(params):
        """
        Transforms the weights into the appropriate broadcasting form for the given 
        number of qubits (FIRST version).
        """
        n_qubits = 10
        params_flat = params.reshape(-1)
        opt_length = 2 ** n_qubits

        if len(params_flat) < opt_length:
            params_flat = np.pad(params_flat, (0, opt_length - len(params_flat)),
                                 mode='constant', constant_values=0.0)
        
        params_alpha = np.array(params_flat) # Convert to Numpy array

        return params


    # BROADCASTING (VERSION #2):
    @staticmethod
    def broadcast_params_V2(params, pad=True):
        """
        Transforms the weights into the appropriate broadcasting form for the given
        number of qubits (SECOND version).
        """
        n_qubits = 10
        params_flat = params.reshape(-1)
        opt_length = 2 ** n_qubits

        # Pad Parameters with Zeros (As Needed):
        if pad is True:
            if len(params_flat) < opt_length:
                params_flat = np.pad(params_flat, (0, opt_length - len(params_flat)),
                                    mode='constant', constant_values=0.0)
        
        params = np.array(params_flat, requires_grad=True) # Convert to Numpy array

        return params


    # PREPARING WEIGHTS (VERSION #1): 
    @staticmethod
    def param_prep_V1(params, active_qubits=None, n_qubits=None,
                      complex=False):
        """
        Prepares weights for passing into QCNN: Transforms the weights into the appropriate
        broadcasting form for the given number of qubits (size of 2^{n_qubits}, pad with zeros).
        Then transforms the parameters to a Torch tensor of type 'complex128' with 
        'requires_grad' set to 'True' (FIRST version).
        """
        # Change based on number of qubits used in QC
        #---------------------------------------
        # Check 'active_qubits' is passed:
        if active_qubits is None:
            active_qubits = 10
        
        # Check 'n_qubits' is passed:
        if n_qubits is None:
            n_qubits = 10
        #---------------------------------------
        params_flat = params.reshape(-1)
        opt_length = 2 ** n_qubits

        if len(params_flat) < opt_length:
            params_flat = np.pad(params_flat, (0, opt_length - len(params_flat)),
                                 mode='constant', constant_values=0.0)
        
        # Convert to the specified dtype:
        if complex == False:
            params_alpha = params_flat.astype(torch.complex128)
        else:
            params_alpha = params_flat

        # Convert to Torch tensor:
        params = torch.tensor(params_alpha, requires_grad=True)

        return params


    # PREPARING WEIGHTS (VERSION #2):
    @staticmethod
    def param_prep_V2(params, active_qubits=None, n_qubits=None,
                      dtype_key=None):
        """
        Prepares weights for passing into QCNN: Transforms the weights into the appropriate
        broadcasting form for the given number of qubits (size of 2^{n_qubits}, pad with zeros).
        Then transforms the parameters to a Torch tensor with the specified dtype and 
        'requires_grad' set to 'True' (SECOND version).
        
        List of potential datatypes (THREE total):
        -> 'complex' : Converts to 'complex128'
        -> 'float64' : Converts to 'float64'
        -> 'default' : No conversion, keeps original type
        """
        # Dictionary for dtype selection
        dtype_dict = {
            'complex': (np.complex128, torch.complex128),
            'float': (np.float64, torch.float64),
            'default': (np.int64, torch.long)
        }

        #-------------------------------------------------
        # Check 'active_qubits' is passed:
        if active_qubits is None:
            active_qubits = 10 # Change as needed for QC
        
        # Check 'n_qubits' is passed:
        if n_qubits is None:
            n_qubits = 10 # Change as needed for QC
        #-------------------------------------------------
            
        params_flat = params.reshape(-1)
        opt_length = 2 ** n_qubits

        # Pad Parameters with Zeros (As Needed):
        if len(params_flat) < opt_length:
            params_flat = np.pad(params_flat, (0, opt_length - len(params_flat)),
                                mode='constant', constant_values=0.0)
        
        # Shape and dtype conversion:
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


    # CIRCUIT DRAWING (VERSION #1):
    @staticmethod
    def draw_qcnn_V1(qc_self, qc_func, params, x,
                     active_qubits=None, n_qubits=None, num_wires=None):
        """
        Draws the corresponding QCNN quantum circuit (FIRST version). Uses qml.draw_mpl().
        """
        #--------------------------------------------
        # Check 'active_qubits' is passed:
        if active_qubits is None:
            # active_qubits = self.active_qubits
            active_qubits = 10
        
        # Check 'n_qubits' is passed:
        if n_qubits is None:
            # n_qubits = self.n_qubits
            n_qubits = 10

        # Check 'num_wires' is passed:
        if num_wires is None:
            num_wires = 2
        #--------------------------------------------
        
        # Initialize Device:
        dev = qml.device("default.qubit", wires=num_wires)
        
        @qml.qnode(dev)
        def circuit_lppc(qc_self, params):
            qc_func(qc_self, params, x, draw=True)
            return [qml.expval(qml.PauliZ(wire)) for wire in range(num_wires)]
        
        # Construct / Plot Figure:
        fig = plt.figure(figsize=(10, 7))
        qml.draw_mpl(circuit_lppc, expansion_strategy="device")(qc_self, params)

        plt.show()


    # CIRCUIT DRAWING (VERSION #2):
    @staticmethod
    def draw_qcnn_V2(qc_self, qc_func, params, x,
                     active_qubits=None, n_qubits=None, num_wires=None, dev=False):
        """
        Draws the corresponding QCNN quantum circuit (SECOND version). Uses qml.draw().
        """
        #--------------------------------------------
        # Check 'active_qubits' is passed:
        if active_qubits is None:
            # active_qubits = self.active_qubits
            active_qubits = 10
        
        # Check 'n_qubits' is passed:
        if n_qubits is None:
            # n_qubits = self.n_qubits
            n_qubits = 10

        # Check 'num_wires' is passed:
        if num_wires is None:
            num_wires = 2
        #--------------------------------------------
        
        if dev is True:
        # Initialize Device:
            device = qml.device("default.qubit", wires=num_wires)
            
            @qml.qnode(device)
            def circuit_lppc(qc_self, params):
                qc_func(qc_self, params, x, draw=True)
                return [qml.expval(qml.PauliZ(wire)) for wire in range(num_wires)]
        
        # Construct / Plot Figure:
        # qcnn_draw = qml.draw(circuit_lppc)
        qc_draw = qml.draw(qc_func)
        print(qc_draw(params, wires=range(num_wires)))


################### MAIN ######################

def main():
    return


if __name__ == "__main__":
    main()
