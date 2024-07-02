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
from scipy.linalg import expm


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
        Transforms the parameters to a Torch tensor of type 'complex128'.
        """
        params_complex = params.astype(np.complex128)
        params = torch.tensor(params_complex)
        return params

    # BROADCASTING WEIGHTS:
    def broadcast_params(self, params):
        """
        Transforms the weights into the appropriate broadcasting form for the given number of qubits.
        """
        n_qubits = self.n_qubits
        params_flat = params.reshape(-1)
        opt_length = 2 ** n_qubits

        if len(params_flat) < opt_length:
            params_flat = np.pad(params_flat, (0, opt_length - len(params_flat)),
                                 mode='constant', constant_values=0.0)
        
        params = np.array(params_flat)
        return params


################### MAIN ######################

def main():
    return


if __name__ == "__main__":
    main()
