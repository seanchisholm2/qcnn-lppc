from setuptools import setup, find_packages

setup(
    name='lppc_qcnn',
    version='0.1.0',
    description='Implementation of a quantum convolutional neural network (QCNN) with the Pennylane QML framework',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Sean Chisholm',
    author_email='seanchisholm2@gmail.com',
    url='https://github.com/seanchisholm2/qcnn-lppc',
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'seaborn',
        'pennylane',
        'numpy',
        'jax',
        'optax',
        'pandas',
        'scikit-learn',
        'torch',
    ],
    extras_require={
        'datetime': ['datetime'],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
