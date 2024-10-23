# chipsff

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
```

## Setup
Clone the repository and install the necessary Python dependencies:
```bash
git clone <repository-url>
cd <repository-folder>
pip install -r requirements.txt
```

## Usage

### 1. Single Material Analysis
To run an analysis on a single material by specifying its JID and calculator type:
```bash
python materials_analyzer.py --jid <JARVIS_ID> --calculator_type <calculator_type>
```

### 2. Batch Processing
To analyze multiple materials at once:
```bash
python materials_analyzer.py --jid_list <JID1> <JID2> --calculator_types <calculator_type1> <calculator_type2>
```

### 3. Interface Analysis
To run an interface analysis between a film and substrate:
```bash
python materials_analyzer.py --film_jid <Film_JID> --substrate_jid <Substrate_JID> --calculator_type <calculator_type>
```

### 4. Thermal and Phonon Analysis
Phonon and thermal properties can be analyzed and plotted using:
```bash
python materials_analyzer.py --jid <JARVIS_ID> --calculator_type <calculator_type>
```

### 5. Vacancy and Surface Energy Analysis
Vacancy and surface energy calculations are automatically handled during the analysis pipeline.

## Functions
- **relax_structure()**: Optimizes the structure using the selected calculator.
- **calculate_formation_energy()**: Computes the formation energy of the relaxed structure.
- **calculate_elastic_tensor()**: Calculates the elastic tensor.
- **calculate_ev_curve()**: Fits an energy-volume curve and computes the bulk modulus.
- **run_phonon_analysis()**: Performs phonon band structure and thermal properties calculations.
- **analyze_defects()**: Analyzes vacancy formation energy.
- **analyze_interfaces()**: Performs interface energy analysis between a film and substrate.

## Example
To run a full analysis on a material with JID 'JVASP-1002' using the `alignn_ff` calculator:
```bash
python materials_analyzer.py --jid JVASP-1002 --calculator_type alignn_ff
```

## Outputs
The results of the calculations are saved in JSON format in the output directory for each material. Key outputs include:
- `job_log.txt`: Log file of the analysis process.
- `POSCAR_*`: Relaxed structure files in VASP POSCAR format.
- `*_results.json`: Final results, including energy, surface energy, vacancy energy, etc.
- Plots of energy-volume curves, phonon bands, thermal expansion, and radial distribution functions (RDF).

## License
This project is licensed under the MIT License.

## Contributing
Feel free to submit pull requests or report issues to improve the repository.

---

## Code Explanation

### Main Dependencies
The following Python libraries are primarily used:
- `ase`: Atomic Simulation Environment, used for handling atomic structures and calculations.
- `jarvis-tools`: Used to access materials data from the JARVIS database.
- `phonopy` and `phono3py`: For phonon and thermal properties calculations.
- `alignn-ff`, `chgnet`, `sevenn`, `mace`: Various force field calculators for materials simulations.

### Core Functions

#### `get_entry(jid)`
Fetches the material data from the JARVIS database using its JID.

#### `collect_data(dft_3d, vacancydb, surface_data)`
Aggregates the DFT, vacancy, and surface energy data from JARVIS for further analysis.

#### `get_vacancy_energy_entry(jid, aggregated_data)`
Fetches the vacancy formation energy entry for a given material based on its JID.

#### `get_surface_energy_entry(jid, aggregated_data)`
Fetches the surface energy entry for a given material based on its JID.

#### `MaterialsAnalyzer`
This is the main class of the repository that handles the entire analysis pipeline for materials. Below are the key methods in the class:

1. **`__init__(...)`**: Initializes the analyzer for a material or interface.
2. **`log_job_info(message, log_file)`**: Logs the analysis steps and results.
3. **`relax_structure()`**: Relaxes the atomic structure and logs the process.
4. **`calculate_formation_energy()`**: Computes the formation energy based on the relaxed structure.
5. **`calculate_elastic_tensor()`**: Calculates the elastic tensor for the material.
6. **`calculate_ev_curve()`**: Fits the energy-volume curve using an equation of state.
7. **`run_phonon_analysis()`**: Runs phonon band structure analysis, density of states, and thermal properties.
8. **`analyze_defects()`**: Analyzes vacancy formation energies.
9. **`analyze_surfaces()`**: Analyzes surface energies for the material.
10. **`analyze_interfaces()`**: Performs interface energy analysis between a film and substrate.
11. **`run_phonon3_analysis()`**: Runs third-order phonon calculations for thermal conductivity.
12. **`general_melter()`**: Runs MD melting and quenching simulations for atomic structures.

### Example Scripts

#### Single Material Analysis
To run the complete analysis pipeline for a single material with a specified calculator:
```bash
python materials_analyzer.py --jid <JARVIS_ID> --calculator_type <calculator_type>
```

#### Multiple Materials Batch Processing
To analyze a list of materials:
```bash
python materials_analyzer.py --jid_list <JID1> <JID2> --calculator_types <calculator_type1> <calculator_type2>
```

#### Interface Analysis
To analyze the interface between two materials:
```bash
python materials_analyzer.py --film_jid <Film_JID> --substrate_jid <Substrate_JID> --calculator_type <calculator_type>
```
