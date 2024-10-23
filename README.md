# chipsff

# Materials Analyzer

## Overview
This repository provides a comprehensive framework for performing materials analysis, including defect and surface energy calculations, vacancy formation, interface analysis, and thermal properties like phonon band structures and elastic tensors. The code supports multiple calculators and workflows that integrate with the JARVIS database and the ASE (Atomic Simulation Environment) to ensure seamless execution of materials calculations.

## Features
- **Structure Relaxation**: Optimizes atomic structures using FIRE optimization.
- **Vacancy and Surface Energy Calculations**: Computes vacancy formation energies and surface energies for specified materials.
- **Phonon Analysis**: Generates phonon band structures, density of states (DOS), and thermal properties using Phonopy.
- **Elastic Properties**: Calculates elastic tensors and bulk modulus.
- **Energy-Volume (E-V) Curve**: Fits the E-V curve using an equation of state (EOS).
- **Thermal Expansion**: Performs thermal expansion analysis using the Quasi-Harmonic Approximation (QHA).
- **Defect and Interface Analysis**: Handles defects and interface calculations for various materials.

## Requirements
The following libraries and tools are required:
- `ase`
- `jarvis-tools`
- `phonopy`
- `phono3py`
- `alignn-ff`
- `chgnet`
- `sevenn`
- `mace`
- `pandas`
- `matplotlib`
- `numpy`
- `plotly`
- `scikit-learn`

You can install all dependencies by running:
```bash
pip install -r requirements.txt

