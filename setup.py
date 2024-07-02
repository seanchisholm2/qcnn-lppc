#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from setuptools import setup, find_packages


# In[ ]:


setup(
    name="lppc_qcnn_package",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pennylane",
        "matplotlib",
        "torch"
    ],
    description="PennyLane Quantum Convolutional Neural Network Package (LPPC)",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/seanchisholm2/qcnn-lppc",
    author="Sean Chisholm",
    author_email="schisholm@college.harvard.edu",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

