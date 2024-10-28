# chipsff

## Overview

The `chipsff` repository provides a comprehensive framework for performing materials analysis, including structure relaxation, defect and surface energy calculations, vacancy formation energies, interface analysis, and thermal properties like phonon band structures and elastic tensors. The code supports multiple calculators, including machine learning force fields (MLFFs), and integrates with the JARVIS database and the Atomic Simulation Environment (ASE) to facilitate various materials simulations and calculations.

## Features

- **Structure Relaxation**: Optimize atomic structures using various calculators and optimization algorithms.
- **Vacancy and Surface Energy Calculations**: Compute vacancy formation energies and surface energies for specified materials.
- **Phonon Analysis**: Generate phonon band structures, density of states (DOS), and thermal properties using Phonopy.
- **Elastic Properties**: Calculate elastic tensors and bulk modulus.
- **Energy-Volume (E-V) Curve**: Fit the E-V curve using an equation of state (EOS) to obtain bulk modulus and equilibrium volume.
- **Thermal Expansion**: Perform thermal expansion analysis using the Quasi-Harmonic Approximation (QHA).
- **Defect and Interface Analysis**: Analyze defects and perform interface calculations between different materials.
- **Molecular Dynamics (MD) Simulations**: Conduct MD simulations to melt and quench structures, and calculate Radial Distribution Functions (RDFs).
- **Support for Multiple Calculators**: Seamlessly switch between different calculators like `alignn_ff`, `chgnet`, `sevenn`, `mace`, `matgl`, etc.

## Requirements

The following libraries and tools are required:

- `python >= 3.6`
- `numpy`
- `matplotlib`
- `pandas`
- `scikit-learn`
- `ase`
- `phonopy`
- `phono3py`
- `jarvis-tools`
- `h5py`
- `plotly`
- `ruamel.yaml`

You can install all dependencies by running:

```bash
pip install -r requirements.txt
```

**Note**: Some calculators may have additional dependencies or require specific versions of libraries. Please refer to their respective documentation for setup instructions.
