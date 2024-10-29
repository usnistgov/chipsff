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
- `ruamel`

You can install all dependencies by running:

```bash
pip install -r requirements.txt
```

## Universal MLFFs Implemented

- `alignn_ff`
- `chgnet`
- `sevenn`
- `mace`
- `matgl`
- `orb`
- `fairchem`

**Note**: Some calculators may have additional dependencies or require specific versions of libraries. Please refer to their respective documentation for setup instructions.

## Usage
The main script `run_chipsff.py` provides a command-line interface to perform various materials analyses.

**1. Single Material Analysis**
To run an analysis on a single material by specifying its JID (JARVIS ID) and calculator type (uMLFF):
```bash
python run_chipsff.py --input_file input.json
```
An example `input.json` file: 
```bash
{
  "jid": "JVASP-1002",
  "calculator_type": "chgnet",
  "chemical_potentials_file": "chemical_potentials.json",
  "properties_to_calculate": [
    "relax_structure",
    "calculate_ev_curve",
    "calculate_formation_energy",
    "calculate_elastic_tensor",
    "run_phonon_analysis",
    "analyze_surfaces",
    "analyze_defects",
    "run_phonon3_analysis",
    "general_melter",
    "calculate_rdf"
  ],
"bulk_relaxation_settings": {
  "filter_type": "ExpCellFilter",
  "relaxation_settings": {
    "fmax": 0.05,
    "steps": 200,
    "constant_volume": false
  }
},
  "phonon_settings": {
    "dim": [2, 2, 2],
    "distance": 0.2
  },
  "use_conventional_cell": true,
  "surface_settings": {
    "indices_list": [
      [0, 1, 0],
      [0,0,1]
    ],
    "layers": 4,
    "vacuum": 18,
    "relaxation_settings": {
      "fmax": 0.05,
      "steps": 200,
      "constant_volume": true
    },
    "filter_type": "ExpCellFilter"
  },
  "defect_settings": {
    "generate_settings": {
      "on_conventional_cell": true,
      "enforce_c_size": 8,
      "extend": 1
    },
    "relaxation_settings": {
      "fmax": 0.05,
      "steps": 200,
      "constant_volume": true
    },
    "filter_type": "ExpCellFilter"
  },
  "phonon3_settings": {
    "dim": [2, 2, 2],
    "distance": 0.2
  },
  "md_settings": {
    "dt": 1,
    "temp0": 35,
    "nsteps0": 10,
    "temp1": 200,
    "nsteps1": 20,
    "taut": 20,
    "min_size": 10.0
  }
}
```
**2. Interface Analysis**
To perform an interface analysis between a film and substrate:
```bash
python run_chipsff.py --input_file interface_input.json
```
An example `interface_input.json` file:
```bash
{
  "film_id": ["JVASP-1002"],
  "substrate_id": ["JVASP-816"],
  "calculator_type": "alignn_ff",
  "chemical_potentials_file": "chemical_potentials.json",
  "film_index": "1_1_0",
  "substrate_index": "1_1_0",
  "properties_to_calculate": [
    "analyze_interfaces"
  ]
}
```
## Key Methods

- `relax_structure()`: Optimizes the atomic structure using the specified calculator and relaxation settings.
- `calculate_formation_energy(relaxed_atoms)`: Computes the formation energy per atom based on the relaxed structure and chemical potentials.
- `calculate_elastic_tensor(relaxed_atoms)`: Calculates the elastic tensor for the relaxed structure.
- `calculate_ev_curve(relaxed_atoms)`: Fits the energy-volume curve using an equation of state to obtain bulk modulus and equilibrium volume.
- `run_phonon_analysis(relaxed_atoms)`: Performs phonon band structure calculations, density of states, and thermal properties using Phonopy.
- `analyze_defects()`: Analyzes vacancy formation energies by generating defects, relaxing them, and calculating formation energies.
- `analyze_surfaces()`: Analyzes surface energies by generating surface structures, relaxing them, and calculating surface energies.
- `run_phonon3_analysis(relaxed_atoms)`: Runs third-order phonon calculations for thermal conductivity using Phono3py.
- `calculate_thermal_expansion(relaxed_atoms)`: Calculates the thermal expansion coefficient using the Quasi-Harmonic Approximation.
- `general_melter(relaxed_atoms)`: Performs MD simulations to melt and quench the structure, then calculates the Radial Distribution Function (RDF).
- `analyze_interfaces()`: Performs interface analysis between film and substrate materials using the `intermat` package.
