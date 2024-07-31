########################################### LOAD_QC_DATA.PY ###########################################

#### ***** IMPORTS / DEPENDENCIES *****:

### *** PLOTTING ***:
import matplotlib.pyplot as plt

### *** PENNYLANE ***:
from pennylane import numpy as np

### *** DATA ***:
from sklearn import datasets
import seaborn as sns
sns.set()
from functools import partial

### *** JAX ***:
import jax;
## JAX CONFIGURATIONS:
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
## OTHER (JAX):
from jax import lax # Dynamic splicing

# ---------------------------------------------------------------
### *** TORCHVISION (FOR DATA):
import torch
# from torchvision import datasets # (TorchVision not in env)
# from torchvision import transforms # (TorchVision not in env)
from torch.utils.data import DataLoader, TensorDataset

### *** TENSORFLOW (FOR DATA) ***:
# import tensorflow as tf
# from tensorflow.keras.datasets import mnist
# ---------------------------------------------------------------

### *** RNG ***:
seed = 0
# Using NumPy (base):
# rng = np.random.default_rng(seed=seed) # ORIGINAL (NumPy)
# Using JAX (base):
rng_jax = jax.random.PRNGKey(seed=seed) # *1* (JAX)
rng_jax_arr = jnp.array(rng_jax) # *2* (JAX)

### *** OTHER ***:
# from glob import glob

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
#                  NEW DATA PREPARATION CLASS
# ============================================================
  
class DataQCNN:
    """
    New class for loading and assessing properties of the MNIST data for quantum 
    convolutional neural networks (class defined just to assess JAX-related data structure issues,
    and is not receiving serious detailed construction or support. Using DataQCNN to quickly write
    functions to assess relevant erros, only if writing the function would be more efficient and
    organized than not).
    """

    # ----------------------------------------------------
    #             DATA ANALYSIS FUNCTIONS (NEW)
    # ----------------------------------------------------

    # ******* DIGITS DATA ANALYSIS *******:
    @staticmethod
    def dataset_structure(package=None):
        """
        Checks shape and type of MNIST data features and labels, and datatype of members (without any
        typecasting or instantiation using ).
        """
        # Load digits dataset:
        digits = datasets.load_digits()
        features, labels = digits.data, digits.target

        # cast features and labels as array if needed
        features_jax = jnp.asarray(features)
        labels_jax = jnp.asarray(labels)

        if package == 'numpy':
            # Print relevant attributes:
            print(f"------------NUMPY------------")
            print(f"___Shape+Type___:")
            print(f"• FEATURES type:  {type(features)}  | shape:  {features.shape}")
            print(f"• LABELS type:  {type(labels)}  | shape:  {labels.shape}")
            print(f"___DataType___:")
            print(f"• FEATURES dtype:  {features.dtype}  | class type:  {type(features[0])}")
            print(f"• LABELS dtype:  {labels.dtype}  | class type:  {type(labels[0])}")
            print(f"___SIZE___:")
            print(f"• FEATURES size:  {features.size}")
            print(f"• LABELS size:  {labels.size}\n")
            print() # For spacing
        elif package == 'jax':
            # Print relevant attributes:
            print(f"------------JAX------------")
            print(f"___Shape+Type___:")
            print(f"• FEATURES type:  {type(features_jax)}  | shape:  {features_jax.shape}")
            print(f"• LABELS type:  {type(labels_jax)}  | shape:  {labels_jax.shape}")
            print(f"___DataType___:")
            print(f"• FEATURES dtype:  {features_jax.dtype}  | class type:  {type(features_jax[0])}")
            print(f"• LABELS dtype:  {labels_jax.dtype}  | class type:  {type(labels_jax[0])}")
            print(f"___SIZE___:")
            print(f"• FEATURES size:  {features_jax.size}")
            print(f"• LABELS size:  {labels_jax.size}\n")
        else:
            raise ValueError("package must be either 'jax' or 'numpy'")
        
    # ----------------------------------------------------
    #             DATA ANALYSIS FUNCTIONS (OLD)
    # ----------------------------------------------------
        
     # ******* JAX LOADING DIGITS DATA (DYNAMIC SLICING) *******:
    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1))
    def load_digits_dynamic_slice(n_train, n_test, rng_jax):
        """
        Returns training and testing data of the digits dataset using jax operations.

        Args:
        'n_train' (int): The number of training samples to select from the dataset.
        'n_test' (int): The number of testing samples to select from the dataset.
        'rng_jax' (jax.random.Generator): A random number generator instance for reproducibility.
        """
        digits = datasets.load_digits()
        features, labels = digits.data, digits.target

        # Features and Labels -> JAX arrays
        features = jnp.asarray(features)
        labels = jnp.asarray(labels)

        # Only use first two classes (0 and 1):
        mask = (labels == 0) | (labels == 1)
        features = features[mask]
        labels = labels[mask]

        # Normalize data
        features = features / jnp.linalg.norm(features, axis=1, keepdims=True)

        # Generate shuffled indices:
        rng_jax, subkey = jax.random.split(rng_jax)
        shuffled_indices = jax.random.permutation(subkey, len(labels))

        n_train = jnp.int64(n_train)
        n_test = jnp.int64(n_test)

        # Subsample train and test split (NEW)
        train_indices = lax.dynamic_slice(shuffled_indices, (0,), (n_train,)) # *2* with JAX
        # test_indices = shuffled_indices[num_train:num_train + num_test] # *1* with JAX
        test_indices = lax.dynamic_slice(shuffled_indices, (n_train,), (n_test,)) # *2* with JAX

        x_train, y_train = features[train_indices], labels[train_indices]
        x_test, y_test = features[test_indices], labels[test_indices]

        return (
            jnp.asarray(x_train),
            jnp.asarray(y_train),
            jnp.asarray(x_test),
            jnp.asarray(y_test),
        )


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

    # ******* PREPARING QC DATASET *******:
    @staticmethod
    def prepare_data():
        """
        Prepares MNIST data for QCNN before converting to JAX and using JAX operations. Loads the 
        data, uses boolean indexing to only train on binary classification values (0 and 1) and also
        normalizes the data.
        """
        digits = datasets.load_digits()
        features, labels = digits.data, digits.target

        # Use NumPy for boolean indexing:
        mask = (labels == 0) | (labels == 1)
        features = features[mask]
        labels = labels[mask]

        # Normalize data
        features = features / np.linalg.norm(features, axis=1).reshape((-1, 1))

        return features, labels

    # ******* LOADING DIGITS DATA *******:
    @staticmethod
    def load_digits_data(n_train, n_test, rng):
        """
        Returns training and testing data of the digits dataset.

        Args:
        -> n_train (int): The number of training samples to select from the dataset.
        -> n_test (int): The number of testing samples to select from the dataset.
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
        train_indices = rng.choice(len(labels), n_train, replace=False)
        test_indices = rng.choice(
            np.setdiff1d(range(len(labels)), train_indices), n_test, replace=False
        )

        x_train, y_train = features[train_indices], labels[train_indices]
        x_test, y_test = features[test_indices], labels[test_indices]

        return (
            jnp.asarray(x_train),
            jnp.asarray(y_train),
            jnp.asarray(x_test),
            jnp.asarray(y_test),
        )
    
    # ******* JAX LOADING DATA (V1) *******:
    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1)) # -> 'num_train' and 'num_test'
    def load_digits_jax_V1(n_train, n_test):
        """
        Returns training and testing data of the digits dataset using JAX. Version 1 using jax, which
        loads the relevant dataset directly within the function.
        """

        digits = datasets.load_digits()
        features, labels = digits.data, digits.target

        # convert features and labels to jax arrays
        features = jnp.asarray(features)
        labels = jnp.asarray(labels)

        # only use first two classes
        mask = (labels == 0) | (labels == 1)  # define mask
        # use mask to filter features and labels
        features = features[mask]
        labels = labels[mask]

        # normalize data
        # features = features / jnp.linalg.norm(features, axis=1, keepdims=True)
        features = features / jnp.linalg.norm(features, axis=1).reshape((-1, 1))

        # ### *** INDICES ***:
        seed = 0
        jax_rng = jax.random.PRNGKey(seed=seed) # JAX rng key
        # split jax key into training and testing keys
        train_key, test_key = jax.random.split(jax_rng)

        n_total = labels.shape[0] # total number of labels

        # random permutation for training indices
        permuted_train = jax.random.permutation(train_key, n_total)
        train_indices = permuted_train[:n_train]

        # Exclude train_indices from the total indices
        res_indices = jnp.setdiff1d(jnp.arange(n_total), train_indices)

        idx_total = res_indices.shape[0] # total number of remaining indices

        # random permutation for testing indices
        permuted_test = jax.random.permutation(test_key, idx_total)
        test_indices = res_indices[permuted_test[:n_test]]

        # split features and labels into training and testing data
        x_train, y_train = features[train_indices], labels[train_indices]
        x_test, y_test = features[test_indices], labels[test_indices]

        return (
            jnp.asarray(x_train),
            jnp.asarray(y_train),
            jnp.asarray(x_test),
            jnp.asarray(y_test),
        )
    
    # ******* JAX LOADING DATA (V2) *******:
    @staticmethod
    def load_digits_jax_V2(n_train, n_test, features, labels):
        """
        Returns training and testing data of the digits dataset using JAX. Version 1 using jax, which
        assumes that the relevant features and labels of the dataset for the qcnn have already been
        instantiated, and are passed to the function as arguments.
        """

        # ### *** INDICES ***:
        seed = 0
        jax_rng = jax.random.PRNGKey(seed=seed) # JAX rng key
        # split jax key into training and testing keys
        train_key, test_key = jax.random.split(jax_rng)

        # n_total = labels.shape[0] # total number of labels
        n_total = 64 # (2^n_qubits, 6-qubit system)

        # random permutation for training indices
        permuted_train = jax.random.permutation(train_key, n_total)
        train_indices = permuted_train[:n_train]

        # Exclude train_indices from the total indices
        res_indices = jnp.setdiff1d(jnp.arange(n_total), train_indices)

        idx_total = res_indices.shape[0] # total number of remaining indices

        # random permutation for testing indices
        permuted_test = jax.random.permutation(test_key, idx_total)
        test_indices = res_indices[permuted_test[:n_test]]

        # split features and labels into training and testing data
        x_train, y_train = features[train_indices], labels[train_indices]
        x_test, y_test = features[test_indices], labels[test_indices]

        return (
            jnp.asarray(x_train),
            jnp.asarray(y_train),
            jnp.asarray(x_test),
            jnp.asarray(y_test),
        )
    
    # ******* JAX LOADING DATA (V3) *******:
    @staticmethod
    def load_digits_jax_V3(n_train, n_test, features, labels):
        """
        Returns training and testing data of the digits dataset using JAX. Version 1 using jax, which
        assumes that the relevant features and labels of the dataset for the qcnn have already been
        instantiated, and are passed to the function as arguments. Unlike V2, this version omits the
        use of jnp.setdiff1d and directly shuffles indices instead.
        """
        # *** INDICES ***:
        seed = 0
        jax_rng = jax.random.PRNGKey(seed=seed) # JAX rng key

        # n_total = labels.shape[0] # total number of labels
        n_total = 64 # (2^n_qubits, 6-qubit system) -> EQUAL TO LENGTH OF 'LABELS'

        # Shuffle indices:
        shuffled_indices = jax.random.permutation(jax_rng, n_total)

        # Split indices for training and testing:
        train_indices = shuffled_indices[:n_train]
        test_indices = shuffled_indices[n_train:n_train + n_test]

        # Split data using indices:
        x_train, y_train = features[train_indices], labels[train_indices]
        x_test, y_test = features[test_indices], labels[test_indices]

        return (
            jnp.asarray(x_train),
            jnp.asarray(y_train),
            jnp.asarray(x_test),
            jnp.asarray(y_test),
        )
    
    # ******* LOADING DIGITS DATA WITH JAX *******:
    @staticmethod
    def load_digits_data_jax(n_train, n_test, features, labels):
        return LoadDataQC.load_digits_jax_V3(n_train, n_test, features, labels)
    
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

    # ******* SAMPLE QC DATA (INOPERATIVE) *******:
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
        # (Note: Uncomment TensorFlow (FOR DATA) in above imports to use 'mnist', as well
        # as the coding block below):
        '''
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        
        # (Note: This function uses TensorFlow to import the MNIST dataset, which is 
        # currently not utilized within the QCCN example notebook. Instead, our 
        # notebook imported the MNIST dataset using TorchVision).
        
        return train_images, train_labels, test_images, test_labels
        '''
        pass
    
    # ******* LOADING MNIST DATA (TORCHVISION) *******:
    @staticmethod
    def load_mnist_torchvision(batch_train, batch_test, root):
        """
        Loads the MNIST dataset and returns training and testing images and 
        labels, using TorchVision.
        """
        # Load MNIST dataset using TorchVision
        # (Note: Uncomment torchvision in above imports to use 'mnist', as well
        # as the coding block below):
        '''
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
        '''
        pass
    
    def load_mnist_torch(batch_train, batch_test):
        """
        Loads the MNIST dataset and returns training and testing images and 
        labels, using Torch only.
        """
        # Define Root:
        root = './mnist_data'

        # Download MNIST dataset from torch
        train_data = torch.hub.load('pytorch/vision', 'mnist', split='train', root=root, download=True)
        test_data = torch.hub.load('pytorch/vision', 'mnist', split='test', root=root, download=True)

        # Extract images and labels from train and test data
        # train_images = train_data.data.unsqueeze(1).float()
        train_images = train_data.data.unsqueeze(1).float() / 255.0 # normalized
        train_labels = train_data.targets
        # test_images = test_data.data.unsqueeze(1).float()
        test_images = test_data.data.unsqueeze(1).float() / 255.0 # normalized
        test_labels = test_data.targets

        # Create TensorDataset and DataLoader
        trainset = TensorDataset(train_images, train_labels)
        testset = TensorDataset(test_images, test_labels)
        trainloader = DataLoader(trainset, batch_size=batch_train, shuffle=True)
        testloader = DataLoader(testset, batch_size=batch_test, shuffle=True)

        # Extract Data from Loaders
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
