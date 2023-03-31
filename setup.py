#!/usr/bin/env python
import setuptools
import os


os.chmod("Pendulum_Identification.py", 0o744)
os.chmod("CartPole_Identification.py", 0o744)
os.chmod("Binary_Classification.py", 0o744)

setuptools.setup(
    name='NodeREN',
    version='1.0',
    url='https://github.com/DecodEPFL/NodeREN',
    license='CC-BY-4.0 License',
    author='Daniele Martinelli',
    author_email='daniele.martinelli@epfl.ch',
    description='Neural Ordinary Differential Equations meet Recurrent Equilibrium Networks',
    packages=setuptools.find_packages(),
    install_requires=['torch>=2.0.0',
                      'torchdiffeq>=0.2.3',
                      'numpy>=1.24.2',
                      'matplotlib>=3.7.1',
                      'scikit-learn>=1.2.2'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.8',
)
