# "lppc-qcnn"

**A repository implementing a quantum convolutional neural network using the PennyLane QML package framework.**

## Overview

The `lppc-qcnn` package provides an implementation of a Quantum Convolutional Neural Network (QCNN) using the PennyLane QML package framework. This model was designed to explore the potential of quantum machine learning with training on large datasets, more specifically those produced by the IceCube Neutrino Observatory. By employing data compression capabilities of quantum systems, the aim is to use this QCNN model to effectively work with the entirety of IceCube's daily data influx, avoiding any need for data reduction techniques. This approach allows for IceCube's data stream to be processed completely and seamlessly, while also making use of the beneficial properties of quantum circuits.

This repository includes a detailed instructional Jupyter notebook in order to guide users through the process of loading and preparing the MNIST dataset, constructing and visualizing the quantum circuit, and training a QCNN with results. This notebook serves as an educational tool for understanding the key concepts of quantum computing, quantum machine learning, and quantum convolutional neural networks.

## Features

- **Quantum Convolutional Neural Network (QCNN):** Implements a QCNN model using PennyLane, with a focus on ultimately using data from the IceCube Neutrino Observatory, but in this case using the MNIST dataset.
- **Instructional Notebook:** Comprehensive Jupyter notebook (`QCNN_Instructional_NB.ipynb`) is provided to demonstrate how to set up, train, and evaluate the QCNN.
- **MNIST Dataset:** The instructional notebook uses the MNIST dataset to demonstrate how the QCNN can be trained using image data commonly used for training other machine learning models, and to show that this QCNN can be adapted for other datasets. The MNIST dataset is a large set of roughly 70,000 grayscaled images of numerical digits that are each 28x28 pixels, although for use in the instructional notebook, only a subsample of the entire MNIST dataset is imported and processed.
- **Environment Setup:** Environment-related files (`environment.yaml` and `requirements.txt`) are included to make sure that all packages dependencies are correctly installed and updated.

## Installation

Please follow the steps below to correctly set up your environment and install the `lppc-qcnn` package:

1. **Cloning the Repository**

   Clone the `qcnn-lppc` repository to your local machine with the following commands:

   ```bash
   git clone https://github.com/seanchisholm2/qcnn-lppc.git
   cd qcnn-lppc
   ```

2. **Setting Up the Environment**

   Create a new environment for the package using the provided `environment.yaml` file:

   ```bash
   conda env create -f environment.yaml
   conda activate qcnn_env_lppc
   ```

   You can install all required package dependencies using `pip` and the provided `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

3. **Installing the `lppc-qcnn` Package**

   The `lppc-qcnn` package can be installed in two different ways:

   **With `setup.py`:**

   Run the following command in the root directory of the project (where the `setup.py` file is located):

   ```bash
   python setup.py install
   ```

   **With `pip`:**

   You could also install the package in editable mode, allowing you to make changes to the base of the code and use them without needing to always reinstall the package:

   ```bash
   pip install -e ./lppc_qcnn
   ```

## Usage

To start using the QCNN, you can open the provided instructional notebook as follows:

```bash
jupyter notebook `QCNN_Instructional_NB.ipynb`
```

This notebook guides you through the process of creating and training the QCNN model, organized into the following sections:

1. **Loading the MNIST Dataset:** Import and prepare the MNIST dataset for quantum embedding.
2. **Constructing the Quantum Circuit:** Set up the QCNN structure using quantum layers and classical operations, and visualize these constructions by making circuit diagrams.
3. **Training MNIST Data on Quantum Circuit:** Train the constructed QCNN model using the MNIST dataset.
4. **Results:** Evaluate the performance of the QCNN and graph the training results.
5. **Appendix:** Additional circuit diagrams and explanations related to the QCNN are also included.

## Background

Quantum computing and quantum machine learning provide an excellent opportunity to be able to apply quantum theory to current classical machine learning models and algorithms as a way to support and enhance the efficiency of large-scale computation. One example of this is the application of random access codes (RACs), which are used when encoding data within messages that are transmitted and received. My research more specifically attempts to tackle this issue in relation to The IceCube Neutrino Observatory. Currently, IceCube produces approximately 1 TB of data per day.

To manage this large data output, triggers inspired by familiar physics models select and reduce certain data to a more manageable level. This, however, leaves IceCube vulnerable to what is known as the “streetlight effect”, which is an observational bias in which new physics is searched for in the best known areas, thereby neglecting areas where we lack familiarity. While classical computing methods have proven themselves invaluable in data processing, such as recent successes of machine learning in IceCube, quantum computing can be seen as a means to analyze more data to avoid this bias. 

## Contributions

This package was completed in association with the Laboratory of Particle Physics and Cosmology (LPPC) at Harvard University. Inspiration for this work was drawn from the following GitHub repository: [Jaybsoni/Quantum-Convolutional-Neural-Networks](https://github.com/Jaybsoni/Quantum-Convolutional-Neural-Networks).

If you would like to contribute to the `lppc-qcnn` project, please fork this repository and submit a pull request. Contributions that improve the code, add new features that boost efficiency, or that enhance the instructional abilities of the notebook's contents are always welcome.

## License

This project is licensed under the Apache-2.0 License. Please view the [LICENSE](LICENSE) file for further details.
