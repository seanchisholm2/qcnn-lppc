########################################## qc_data.py ############################################

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


################### MNIST DATASET CLASS ######################


# CLASS IMPORTS ("gellmann_ops.py"):
from .gellmann_ops import GellMannOps as gell_ops

class DataLPPC:
    """
    Class for loading and processing MNIST data for quantum convolutional neural networks.
    """
    def __init__(self):
        self.gell_ops = gell_ops() # Initialize GellMannOps to access its variables
        self.n_qubits = gell_ops.n_qubits
        self.num_active_qubits = gell_ops.num_active_qubits
        self.num_qubits = gell_ops.num_qubits
        self.active_qubits = gell_ops.active_qubits

    
    # LOADING MNIST DATA FUNCTION (TENSORFLOW):
    @staticmethod
    def load_mnist_tf():
        """
        Loads the MNIST dataset and returns training and testing images and 
        labels, using TensorFlow.
        """
        # Load MNIST dataset using TensorFlow:
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        
        # *Note*: This function uses TensorFlow to import the MNIST dataset, which is 
        # currently not utilized within the QCCN example notebook. Instead, our 
        # notebook imported the MNIST dataset using TorchVision.
        
        return train_images, train_labels, test_images, test_labels

    
    # LOADING MNIST DATA FUNCTION (TORCHVISION):
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

    
    # REDUCING IMPORTED MNIST DATA FUNCTION:
    @staticmethod
    def mnist_reduce(train_images, train_labels, test_images, test_labels,
                     n_train=500, n_test=100):
        """
        Reduces the dataset size and converts it from multi-classification to binary 
        classification (classes: 0, 1).
        """
        ############# Multi-Classification -> Binary Classification ################
        # Select Indices for Classes 0 and 1:
        train_idx = np.append(np.where(train_labels.numpy() == 0)[0],
                          np.where(train_labels.numpy() == 1)[0]) # <- In 'train_labels'

        test_idx = np.append(np.where(test_labels.numpy() == 0)[0],
                         np.where(test_labels.numpy() == 1)[0]) # <- In 'test_labels'

        # Convert indices to PyTorch Tensors:
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

    
    # FLATTENING IMPORTED MNIST DATA FUNCTION:
    @staticmethod
    def mnist_flatten(train_images, test_images):
        """
        Flattens the MNIST images.
        """
        # Flatten images:
        train_images = train_images.reshape(train_images.shape[0], -1)
        test_images = test_images.reshape(test_images.shape[0], -1)

        return train_images, test_images

    
    # PADDING TRAINING AND TESTING DATA FUNCTION:
    @staticmethod
    def mnist_padding(train_images, train_labels, test_images, test_labels, n_qubits=10):
        """
        Pads MNIST train and test images to the desired shape and returns the padded datasets.
        """
        x_train = np.array(train_images)
        x_test = np.array(test_images)

        y_train = np.array(train_labels)
        y_test = np.array(test_labels)

        # Pad x_train and x_test with zeros to desired shape (, 1024):
        x_train = np.pad(x_train, ((0, 0), (0, (2**n_qubits) - x_train.shape[1])), mode='constant')
        x_test = np.pad(x_test, ((0, 0), (0, (2**n_qubits) - x_test.shape[1])), mode='constant')

        return x_train, y_train, x_test, y_test


################### MAIN ######################

def main():
    return


if __name__ == "__main__":
    main()
