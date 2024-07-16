########################################## gellmann_ops.py ############################################

### IMPORTS / DEPENDENCIES:

# PennyLane:
import pennylane as qml
from pennylane import numpy as np

# TorchVision:
import torch
# from torchvision import datasets, transforms  # NOT ACCESSED
# from torch.utils.data import DataLoader  # NOT ACCESSED

# TensorFlow:
# import tensorflow as tf  # NOT ACCESSED
# from tensorflow.keras.datasets import mnist  # NOT ACCESSED

# OTHER
# *1* Scipy:
from scipy.linalg import expm


################### GELL MANN MATRIX OPERATION CLASS ######################


class GellMannOps:
    """
    Class for generating Gell-Mann matrices and related operations.
    """
    # (Note: Modify 'qubit_config=' line to change qubit configuration across QCNN)
    def __init__(self):
        # QUBITS (ONLY):
        self.n_qubits = self.qubit_select(self, qubit_config="mnist")
        # ACTIVE QUBITS (ONLY):
        self.active_qubits = self.qubit_select(self, qubit_config="mnist")
        # WIRES:
        self.num_wires = 2 # For QCNN Drawings


    # QUBIT AND ACTIVE QUBIT NUMBER SELECTION:
    def qubit_select(self, qubit_config=None, qubit_list=False):
        """
        Selects the number of qubits and active qubits used in QCNN based on string value passed
        for 'qubit_config'. Lists all available selections for number of qubits and active qubits when
        setting 'qubit_list=True' (Note: setting 'qubit_list=True' returns null values).

        Available Qubit Configurations:
        qubit_options = {
            "test + Integer": "'Integer' Number of Qubits/Active Qubits
            "lppc": 4 Qubits/Active Qubits (same as "test2")
            "mnist": 10 Qubits/Active Qubits
            "nullconfig": No Qubits/Active Qubits Defined (Not Defaulted)
        }
        """
        # List of Available Qubit Configurations (THREE total):
        # (Note: rename )
        qubit_options = {
            "test2": (2, 2),
            "test4": (4, 4),
            "test6": (6, 6),
            "test8": (8, 8),
            "test3": (3, 3),
            "test9": (9, 9),
            "test12": (12, 12),
            "lppc": (4, 4), # SAME AS "test2"
            "mnist": (10, 10),
            "nullconfig": (None, None) # NULL QUBITS (as needed)
        }
        
        # Check Qubit Configuration Type:
        if not isinstance(qubit_config, str) and qubit_config is not None:
            raise TypeError("qubit_config must be a string. Set 'qubit_list=True' to view available options.")
        
        # List Available Configurations:
        if qubit_list is True:
            print("Available 'qubit_config' Selections:")
            for key, (n_qubits, active_qubits) in qubit_options.items():
                print(f"{key}: n_qubits = {n_qubits}, active_qubits = {active_qubits} (list of length {active_qubits})")
            return None, None  # Ensure it returns a tuple to avoid errors
        
        # Check if Qubit Configuration is Available for Selection:
        if qubit_config not in qubit_options:
            if qubit_config is None:
                qubit_config = "mnist" # Default to 10-qubit config
            else:
                raise ValueError(f"Invalid qubit_config: {qubit_config}. Set 'qubit_list=True' to view available options.")
        
        # Define Configuration:
        n_qubits, active_qubits = qubit_options[qubit_config]
        active_qubits = list(range(active_qubits)) # Convert 'active_qubits' to list

        return n_qubits, active_qubits

    # BASIS MATRIX:
    def b_mat(self, i, j, n):
        """
        Generates an n x n matrix of 0s with the i,j th entry is a one.
        This is the i,j th basis vector on the space of n x n real matrices.
        Returns np.array of floats, shape (n,n).

        Parameters:
        -> param 'i': int, row index (must be < n)
        -> param 'j': int, column index (must be < n)
        -> param 'n': int, dimension of the matrices
        """
        basis_matrix = np.zeros((n, n), dtype=np.float64) # ORIGINAL: dtype=np.float32
        basis_matrix[i, j] = 1.0
        return basis_matrix


    # GELL-MANN MATRICES:
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


    # CONVOLUTIONAL OPERATOR:
    def get_conv_op(self, mats, params):
        """
        Parametrizes the convolutional operator according to Gell-Mann matrices scaled by 
        trainable parameters, this method generates the relevant applicable operator.
        """
        final = np.zeros(mats[0].shape, dtype=np.complex128)
        for mat, param in zip(mats, params):
            final += param * mat
        return expm(complex(0, -1) * final)


    # CONTROLLED POOL OPERATOR:
    def controlled_pool(self, mat):
        """
        Generates the matrix corresponding the controlled - mat operator. Inputs Numpy array,
        shape (2x2) for the controlled operator and returns the final controlled-mat operator.
        """
        i_hat = np.array([[1.0, 0.0], [0.0, 0.0]])
        j_hat = np.array([[0.0, 0.0], [0.0, 1.0]])
        identity = i_hat + j_hat
        return np.kron(i_hat, identity) + np.kron(j_hat, mat)
    

    # GENERAL ROTATION OPERATOR:
    def G_Rot(self, params, wire):
        """General Rotation Gate to a given Qubit."""
        qml.Rot(params[0], params[1], params[2], wire=wire)
        # return qml.expval(qml.PauliZ(wire))


################### PARAMETER OPERATIONS CLASS ######################
        
        
class ParamOps(GellMannOps):
    """
    Class for handling parameter transformations and broadcasting, dependent on GellMannOps.
    """
    def __init__(self):
        super().__init__()
        # GELLMANNOPS:
        self.gell_ops = GellMannOps() # Initialize 'GellMannOps' to access variables
        # QUBITS:
        self.n_qubits = self.gell_ops.n_qubits
        # ACTIVE QUBITS:
        self.active_qubits = self.gell_ops.active_qubits
        # WIRES:
        self.num_wires = self.gell_ops.num_wires

    # TYPECASTING WEIGHTS FUNCTION:
    def transform_weights(self, params):
        """
        Transforms the parameters to a Torch tensor of type 'complex128' with 
        'requires_grad' set to 'True'.
        """
        params_complex = params.astype(np.complex128) # convert to type 'complex128'
        params_alpha = torch.tensor(params_complex, requires_grad=True) # convert to Torch tensor
        params = np.array(params_alpha) # Convert to Numpy array

        return params
    

    # BROADCASTING WEIGHTS FUNCTION (VERSION #1):
    def broadcast_weights_V1(self, params, n_qubits=None):
        """
        Transforms the weights into the appropriate broadcasting form for the given 
        number of qubits (VERSION #1).
        """
        # QUBIT CHECK:
        #-------------------------------
        # Check 'n_qubits' is passed:
        if n_qubits is None:
            # FOR RUNNING:
            n_qubits = self.n_qubits
            # n_qubits = 10
        #-------------------------------
        
        params_flat = params.reshape(-1)
        opt_length = (2**n_qubits)

        if len(params_flat) < opt_length:
            params_flat = np.pad(params_flat, (0, opt_length - len(params_flat)),
                                 mode='constant', constant_values=0.0)
        
        params_alpha = np.array(params_flat) # Convert to Numpy array

        return params


    # BROADCASTING WEIGHTS FUNCTION (VERSION #2; CURRENT VERSION):
    def broadcast_weights_V2(self, params, n_qubits=None, pad=True):
        """
        Transforms the weights into the appropriate broadcasting form for the given
        number of qubits, and includes an optional padding feature (VERSION #2).
        """
            
        params_flat = params.reshape(-1)
        opt_length = (2**n_qubits)

        # Pad Parameters with Zeros (As Needed):
        if pad is True:
            if len(params_flat) < opt_length:
                params_flat = np.pad(params_flat, (0, opt_length - len(params_flat)),
                                    mode='constant', constant_values=0.0)
        
        params = np.array(params_flat, requires_grad=True) # Convert to Numpy array

        return params
    

    # BROADCASTING WEIGHTS FUNCTION (VERSION #3; CURRENT VERSION):
    def broadcast_weights_V3(self, params,
                             n_qubits=None):
        """
        Transforms the weights into the appropriate broadcasting form for the given
        number of qubits (VERSION #3; CURRENT VERSION).
        """
        # QUBIT CHECK:
        #-------------------------------
        # Check 'n_qubits' is passed:
        if n_qubits is None:
            # FOR RUNNING:
            n_qubits = self.n_qubits
            # n_qubits = 10
        #-------------------------------
            
        params_flat = params.reshape(-1)
        # params = np.array(params, requires_grad=True)
        params = np.array(params_flat, requires_grad=True) # Convert to Numpy array

        return params


    # PREPARING WEIGHTS FUNCTION (VERSION #1): 
    def prep_weights_V1(self, params, n_qubits=None,
                      complex=False):
        """
        Prepares weights for passing into QCNN: Transforms the weights into the appropriate
        broadcasting form for the given number of qubits (size of 2^{n_qubits}, pad with zeros).
        Then transforms the parameters to a Torch tensor of type 'complex128' with 
        'requires_grad' set to 'True' (VERSION #1).
        """
        # QUBIT CHECK:
        #-------------------------------
        # Check 'n_qubits' is passed:
        if n_qubits is None:
            # FOR RUNNING:
            n_qubits = self.n_qubits
            # n_qubits = 10
        #-------------------------------

        params_flat = params.reshape(-1)
        opt_length = (2**n_qubits)

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


    # PREPARING WEIGHTS FUNCTION (VERSION #2; CURRENT VERSION):
    def prep_weights_V2(self, params, n_qubits=None,
                      dtype_key=None):
        """
        Prepares weights for passing into QCNN: Transforms the weights into the appropriate
        broadcasting form for the given number of qubits (size of 2^{n_qubits}, pad with zeros).
        Then transforms the parameters to a Torch tensor with the specified dtype and 
        'requires_grad' set to 'True' (VERSION #2).
        
        List of Available Datatypes (THREE total):
        -> 'complex' : Converts to 'complex128'
        -> 'float' : Converts to 'float64'
        -> 'default' : No conversion, keeps original type
        """
        # QUBIT CHECK:
        #-------------------------------
        # Check 'n_qubits' is passed:
        if n_qubits is None:
            # FOR RUNNING:
            n_qubits = self.n_qubits
            # n_qubits = 10
        #-------------------------------

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
    

    # ******* UPDATED WEIGHTS TRANSFORMATION VERSION(S) *******


    # UPDATED QUBIT SELECTION FUNCTION:
    def qubit_select_lppc(self):
        """
        Returns the most recent version of the QUBIT SELECTION function used in the 
        QCNN with relevant and appropriate parameters passed.
        """
        # Return Qubit Selection ('qubit_select') with appropriate arguments:
        return self.qubit_select(self, qubit_config='mnist', list=False,
                    num_q=True, num_active_q=True)
    

    # UPDATED TRANSFORMING WEIGHTS FUNCTION:
    def transform_weights_lppc(self, *args, **kwargs):
        """
        Returns the most recent version of the TRANSFORMING WEIGHTS function 
        used in the QCNN.
        """
        # Return Current Transform Weights ('transform_weights') with appropriate arguments:
        return self.transform_weights(self, *args, **kwargs)


    # UPDATED BROADCASTING WEIGHTS FUNCTION:
    def broadcast_weights_lppc(self, *args, **kwargs):
        """
        Returns the most recent version of the BROADCASTING WEIGHTS function 
        used in the QCNN (CURRENT VERSION: V3).
        """
        # Return Current Broadcast Weights ('broadcast_weights_V3') with appropriate arguments:
        return self.broadcast_weights_V3(self, *args, **kwargs)

    
    # UPDATED PREPARING WEIGHTS FUNCTION:
    def prep_weights_lppc(self, *args, **kwargs):
        """
        Returns the most recent version of the PREPARING WEIGHTS function 
        used in the QCNN (CURRENT VERSION: V2).
        """
        # Return Current Prepare Weights ('prep_weights_V2') with appropriate arguments:
        return self.prep_weights_V2(self, *args, **kwargs)


################### MAIN ######################


def main():
    return


if __name__ == "__main__":
    main()
