########################################### LOAD_QC_DATA.PY ###########################################

### ***** IMPORTS / DEPENDENCIES *****:

## PLOTTING:
import matplotlib as mpl # (NOT ACCESSED)
import matplotlib.pyplot as plt

## PENNYLANE:
from pennylane import numpy as np

## DATA:
from sklearn import datasets
import seaborn as sns

## JAX:
import jax;
## JAX CONFIGURATIONS:
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.experimental.sparse as jsp # (NOT ACCESSED)
import jax.scipy.linalg as jsl # (NOT ACCESSED)

## OTHER:
sns.set()
seed = 0
rng = np.random.default_rng(seed=seed)

## TORCHVISION (FOR DATA):
import torch
# from torchvision import transforms
# from torchvision import datasets
# from torch.utils.data import DataLoader

## TENSORFLOW (FOR DATA):
# import tensorflow as tf
# from tensorflow.keras.datasets import mnist

### ***** PACKAGE(S) *****:
# ************************************************************************************
# OPERATORS.PY:
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
#                NEW (MNIST) DATA LOADING CLASS
# ============================================================
 

class LoadDataQC:
    """
    New class for loading and processing MNIST data for quantum convolutional neural networks, as
    well as generating random sample data for use in QCNN.
    """
    # ***** IMPORT LoadDataLPPC() CLASS *****:
    # data_lppc = LoadDataLPPC()

    # ----------------------------------------------------
    #     DATA AND LOADING FUNCTIONS (NEW/ESSENTIAL)
    # ----------------------------------------------------

    # ******* NEW LOADING DIGITS DATA *******:
    @staticmethod
    def load_digits_data(num_train, num_test, rng):
        """
        Returns training and testing data of the digits dataset.

        Args:
        -> num_train (int): The number of training samples to select from the dataset.
        -> num_test (int): The number of testing samples to select from the dataset.
        -> rng (numpy.random.Generator): A random number generator instance for reproducibility.
        """
        digits = datasets.load_digits()
        features, labels = digits.data, digits.target

        # only use first two classes
        features = features[np.where((labels == 0) | (labels == 1))]
        labels = labels[np.where((labels == 0) | (labels == 1))]

        # normalize data
        features = features / np.linalg.norm(features, axis=1).reshape((-1, 1))

        # subsample train and test split
        train_indices = rng.choice(len(labels), num_train, replace=False)
        test_indices = rng.choice(
            np.setdiff1d(range(len(labels)), train_indices), num_test, replace=False
        )

        x_train, y_train = features[train_indices], labels[train_indices]
        x_test, y_test = features[test_indices], labels[test_indices]

        return (
            jnp.asarray(x_train),
            jnp.asarray(y_train),
            jnp.asarray(x_test),
            jnp.asarray(y_test),
        )
    
    # ******* VISUALIZING (DIGITS) DATA *******:
    @staticmethod
    def draw_mnist_data():
        """
        Loads the MNIST digits dataset, filters the images and labels for digits 0 and 1,
        and displays the first 12 images in a 1x12 grid.
        """
        digits = datasets.load_digits()
        images, labels = digits.data, digits.target

        images = images[np.where((labels == 0) | (labels == 1))]
        labels = labels[np.where((labels == 0) | (labels == 1))]

        fig, axes = plt.subplots(nrows=1, ncols=12, figsize=(3, 1))

        for i, ax in enumerate(axes.flatten()):
            ax.imshow(images[i].reshape((8, 8)), cmap="gray")
            ax.axis("off")

        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()

    # ******* SAMPLE QUANTUM CIRCUIT DATA *******:
    @staticmethod
    def sample_qcdata():
        """
        Generates sample data for QCNN.
        """
        # TO-DO: Implement this function

        return None


# ============================================================
#              (MNIST) DATA LOADING CLASS (LPPC)
# ============================================================


class LoadDataLPPC:
    """
    Class for loading and processing MNIST data for quantum convolutional neural networks. Class
    functions originating from original QCNN package (LPPC).
    """

    # ----------------------------------------------------
    # ORIGINAL QC AND MNIST DATA LOADING FUNCTIONS (LPPC)
    # ----------------------------------------------------
    
    # ******* LOADING MNIST DATA (TENSORFLOW) *******:
    @staticmethod
    def load_mnist_tf():
        """
        Loads the MNIST dataset and returns training and testing images and 
        labels, using TensorFlow.
        """
        # Load MNIST dataset using TensorFlow
        # (Note: Uncomment TensorFlow (FOR DATA) in above imports to use 'mnist'):
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        
        # (Note: This function uses TensorFlow to import the MNIST dataset, which is 
        # currently not utilized within the QCCN example notebook. Instead, our 
        # notebook imported the MNIST dataset using TorchVision).
        
        return train_images, train_labels, test_images, test_labels
    
    # ******* LOADING MNIST DATA (TORCHVISION) *******:
    @staticmethod
    def load_mnist_torch(batch_train, batch_test, root):
        """
        Loads the MNIST dataset and returns training and testing images and 
        labels, using TorchVision.
        """
        # Define Tensor Transform:
        transform = transforms.Compose([transforms.ToTensor()])

        # Download and Load TRAIN Data:
        trainset = datasets.MNIST(root=root, train=True, transform=transform, download=True)
        trainloader = DataLoader(trainset, batch_size=batch_train, shuffle=True)

        # Download and Load TEST Data:
        testset = datasets.MNIST(root=root, train=False, transform=transform, download=True)
        testloader = DataLoader(testset, batch_size=batch_test, shuffle=True)

        # Extract Data from Loaders:
        train_images, train_labels = next(iter(trainloader))
        test_images, test_labels = next(iter(testloader))
        
        return train_images, train_labels, test_images, test_labels

    # ******* REDUCING IMPORTED MNIST DATA *******:
    @staticmethod
    def mnist_reduce(train_images, train_labels, test_images, test_labels):
        """
        Reduces the dataset size and converts it from multi-classification to binary 
        classification (Note: available class selections are: 0, 1; include 'n_train' and
        'n_test' in arguments as needed).
        """
        # n_train = 500 # Sample TRAIN Batch Size
        # n_test = 100 # Sample TEST Batch Size

        ############### Multi-Classification -> Binary Classification ###############
        # Select Indices for Classes 0 and 1:
        train_idx = np.append(np.where(train_labels.numpy() == 0)[0],
                          np.where(train_labels.numpy() == 1)[0]) # <- In 'train_labels'

        test_idx = np.append(np.where(test_labels.numpy() == 0)[0],
                         np.where(test_labels.numpy() == 1)[0]) # <- In 'test_labels'


        # Convert Indices to PyTorch Tensors:
        train_idx = torch.tensor(train_idx, dtype=torch.long)
        test_idx = torch.tensor(test_idx, dtype=torch.long)

        # Filter Training and Testing Data:
        train_images = train_images[train_idx]
        test_images = test_images[test_idx]

        # Filter Training and Testing Labels:
        train_labels = train_labels[train_idx]
        test_labels = test_labels[test_idx]

        # Convert Training and Testing Data and labels to type 'float64':
        train_images = train_images.type(torch.float64)
        test_images = test_images.type(torch.float64)
        train_labels = train_labels.type(torch.float64)
        test_labels = test_labels.type(torch.float64)

        return train_images, train_labels, test_images, test_labels
    
    # ******* FLATTENING IMPORTED MNIST DATA *******:
    @staticmethod
    def mnist_flatten(train_images, test_images):
        """
        Flattens the MNIST images.
        """
        # Flatten Images:
        train_images = train_images.reshape(train_images.shape[0], -1)
        test_images = test_images.reshape(test_images.shape[0], -1)

        return train_images, test_images
    
    # ******* PADDING TRAINING AND TESTING DATA *******:
    @staticmethod
    def mnist_padding(train_images, train_labels, test_images, test_labels,
                      n_qubits=None, check_qubits=True):
        """
        Pads MNIST train and test images to the desired shape and returns the padded datasets.
        """
        # Check Qubit Configuration:
        if check_qubits is True:
            # Number of Qubits:
            if n_qubits is None:
                # n_qubits = 2
                # n_qubits = 3
                # n_qubits = 4
                # n_qubits = 10
                n_qubits = 6 

        # Convert Data to Numpy Arrays:
        x_train = np.array(train_images)
        x_test = np.array(test_images)

        # Convert Labels to Numpy Arrays:
        y_train = np.array(train_labels)
        y_test = np.array(test_labels)

        # Pad 'x_train' and 'x_test' with zeros to desired shape (, (2**n_qubits)):
        x_train = np.pad(x_train, ((0, 0), (0, (2**n_qubits) - x_train.shape[1])), mode='constant')
        x_test = np.pad(x_test, ((0, 0), (0, (2**n_qubits) - x_test.shape[1])), mode='constant')

        return x_train, y_train, x_test, y_test


# **************************************************************************************************
#                                                MAIN
# **************************************************************************************************


def main():
    return


if __name__ == "__main__":
    main()
