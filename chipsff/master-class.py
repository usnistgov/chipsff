import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from ase.eos import EquationOfState
from ase import Atoms as AseAtoms
from ase.units import kJ
from ase.constraints import ExpCellFilter
from ase.optimize.fire import FIRE
import ase.units
from jarvis.db.figshare import get_jid_data, data, get_request_data
from jarvis.core.atoms import Atoms, ase_to_atoms
from jarvis.io.vasp.inputs import Poscar
from jarvis.core.kpoints import Kpoints3D as Kpoints
from jarvis.analysis.defects.vacancy import Vacancy
from jarvis.analysis.defects.surface import Surface

from phonopy import Phonopy, PhonopyQHA
from phonopy.file_IO import write_FORCE_CONSTANTS
from phonopy.structure.atoms import Atoms as PhonopyAtoms
from phono3py import Phono3py

from alignn.ff.ff import AlignnAtomwiseCalculator, default_path
from matgl.ext.ase import M3GNetCalculator
from chgnet.model.dynamics import CHGNetCalculator
from sevenn.sevennet_calculator import SevenNetCalculator
from mace.calculators import mace_mp
import elastic
from elastic import get_elementary_deformations, get_elastic_tensor
import pandas as pd
import h5py
import shutil
import glob
import io
import contextlib
import re

import plotly.express as px

from sklearn.metrics import mean_absolute_error, r2_score
from matplotlib.gridspec import GridSpec

dft_3d = data("dft_3d")
vacancydb = data("vacancydb")
surf_url = "https://figshare.com/ndownloader/files/46355689"
surface_data = get_request_data(js_tag="surface_db_dd.json", url=surf_url)

# Step 1: Define get_entry function to retrieve JID entry
def get_entry(jid):
    for entry in dft_3d:
        if entry["jid"] == jid:
            return entry
    raise ValueError(f"JID {jid} not found in the database")

# Step 2: Collect data by combining defect and surface info
def collect_data(dft_3d, vacancydb, surface_data):
    defect_ids = list(set([entry["jid"] for entry in vacancydb]))
    surf_ids = list(set([entry["name"].split("Surface-")[1].split("_miller_")[0] for entry in surface_data]))

    aggregated_data = []
    for entry in dft_3d:
        tmp = entry
        tmp["vacancy"] = {}
        tmp["surface"] = {}

        # Check if the entry is in the defect dataset
        if entry["jid"] in defect_ids:
            for vac_entry in vacancydb:
                if entry["jid"] == vac_entry["jid"]:
                    tmp["vacancy"].setdefault(
                        vac_entry["id"].split("_")[0] + "_" + vac_entry["id"].split("_")[1],
                        vac_entry["ef"]
                    )

        # Check if the entry is in the surface dataset
        if entry["jid"] in surf_ids:
            for surf_entry in surface_data:
                jid = surf_entry["name"].split("Surface-")[1].split("_miller_")[0]
                if entry["jid"] == jid:
                    tmp["surface"].setdefault(
                        "_".join(surf_entry["name"].split("_thickness")[0].split("_")[0:5]),
                        surf_entry["surf_en"]
                    )

        aggregated_data.append(tmp)
    
    return aggregated_data

def get_vacancy_energy_entry(jid, aggregated_data):
    """
    Retrieve the vacancy formation energy entry (vac_en_entry) for a given jid.
    
    Parameters:
    jid (str): The JID of the material.
    aggregated_data (list): The aggregated data containing vacancy and surface information.
    
    Returns:
    dict: A dictionary containing the vacancy formation energy entry and corresponding symbol.
    """
    for entry in aggregated_data:
        if entry["jid"] == jid:
            vacancy_data = entry.get("vacancy", {})
            if vacancy_data:
                return [{"symbol": key, "vac_en_entry": value} for key, value in vacancy_data.items()]
            else:
                return f"No vacancy data found for JID {jid}"
    return f"JID {jid} not found in the data."
    

def get_surface_energy_entry(jid, aggregated_data):
    """
    Retrieve the surface energy entry (surf_en_entry) for a given jid.

    Parameters:
    jid (str): The JID of the material.
    aggregated_data (list): The aggregated data containing vacancy and surface information.

    Returns:
    list: A list of dictionaries containing the surface energy entry and corresponding name.
    """
    for entry in aggregated_data:
        if entry["jid"] == jid:
            surface_data = entry.get("surface", {})
            if surface_data:
                # Prepend 'Surface-JVASP-<jid>_' to the key for correct matching
                return [{"name": f"{key}", "surf_en_entry": value} for key, value in surface_data.items()]
            else:
                return f"No surface data found for JID {jid}"
    return f"JID {jid} not found in the data."

# Utility functions
def log_job_info(message, log_file):
    with open(log_file, "a") as f:
        f.write(message + "\n")
    print(message)


def save_dict_to_json(data_dict, filename):
    with open(filename, "w") as f:
        json.dump(data_dict, f, indent=4)


def setup_calculator(calculator_type):
    if calculator_type == "matgl":
        pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        return M3GNetCalculator(pot, compute_stress=True, stress_weight=0.01)
    elif calculator_type == "alignn_ff":
        model_path = default_path()
        return AlignnAtomwiseCalculator(
            path=model_path,
            stress_wt=0.1,
            force_mult_natoms=True,
            force_multiplier=1,
            modl_filename="best_model.pt",
        )
    elif calculator_type == "chgnet":
        return CHGNetCalculator()
    elif calculator_type == "mace":
        return mace_mp()
    elif calculator_type == "sevennet":
        checkpoint_path = "SevenNet/pretrained_potentials/SevenNet_0__11July2024/checkpoint_sevennet_0.pth"
        return SevenNetCalculator(checkpoint_path, device="cpu")
    else:
        raise ValueError("Unsupported calculator type")


class MaterialsAnalyzer:
    def __init__(self, jid, calculator_type, chemical_potentials_json):
        self.jid = jid
        self.reference_data = get_entry(jid)
        self.calculator_type = calculator_type
        self.output_dir = f"{jid}_{calculator_type}"
        os.makedirs(self.output_dir, exist_ok=True)
        self.log_file = os.path.join(
            self.output_dir, f"{jid}_{calculator_type}_job_log.txt"
        )
        self.job_info = {"jid": jid, "calculator_type": calculator_type}
        self.atoms = self.get_atoms(jid)#.get_conventional_atoms
        self.calculator = self.setup_calculator()
        self.chemical_potentials = json.loads(
            chemical_potentials_json
        )  # Load chemical potentials from the provided JSON

    def log(self, message):
        log_job_info(message, self.log_file)

    def get_atoms(self, jid):
        dat = get_jid_data(jid=jid, dataset="dft_3d")
        return Atoms.from_dict(dat["atoms"])

    def setup_calculator(self):
        calc = setup_calculator(self.calculator_type)
        self.log(f"Using calculator: {self.calculator_type}")
        return calc

    def capture_fire_output(self, ase_atoms, fmax, steps):
        """Capture the output of the FIRE optimizer."""
        log_stream = io.StringIO()
        with contextlib.redirect_stdout(log_stream):
            dyn = FIRE(ase_atoms)
            dyn.run(fmax=fmax, steps=steps)
        output = log_stream.getvalue().strip()

        final_energy = None
        if output:
            last_line = output.split("\n")[-1]
            match = re.search(r"FIRE:\s+\d+\s+\d+:\d+:\d+\s+(-?\d+\.\d+)", last_line)
            if match:
                final_energy = float(match.group(1))

        return final_energy, dyn.nsteps

    def relax_structure(self):
        """Perform structure relaxation and log the process."""
        self.log(f"Starting relaxation for {self.jid}")

        # Convert atoms to ASE format and assign the calculator
        ase_atoms = self.atoms.ase_converter()
        ase_atoms.calc = self.calculator
        ase_atoms = ExpCellFilter(ase_atoms)

        # Run FIRE optimizer and capture the output
        final_energy, nsteps = self.capture_fire_output(ase_atoms, fmax=0.05, steps=200)
        relaxed_atoms = ase_to_atoms(ase_atoms.atoms)
        converged = nsteps < 200

        # Log the final energy and relaxation status
        self.log(f"Final energy of FIRE optimization for structure: {final_energy}")
        self.log(
            f"Relaxation {'converged' if converged else 'did not converge'} within 200 steps."
        )

        # Update job info and save the relaxed structure
        self.job_info["relaxed_atoms"] = relaxed_atoms.to_dict()
        self.job_info["final_energy_structure"] = final_energy
        self.job_info["converged"] = converged
        self.log(f"Relaxed structure: {relaxed_atoms.to_dict()}")
        save_dict_to_json(self.job_info, self.get_job_info_filename())

        return relaxed_atoms

    def calculate_formation_energy(self, relaxed_atoms):
        """
        Calculate the formation energy per atom using the equilibrium energy and chemical potentials.
        """
        e0 = self.job_info["equilibrium_energy"]
        composition = relaxed_atoms.composition.to_dict()
        chemical_potentials = (
            self.chemical_potentials
        )  # Use the loaded chemical potentials from the class initialization

        total_energy = e0
        for element, amount in composition.items():
            # Lookup chemical potential for each element
            chemical_potential = chemical_potentials.get(element, {}).get(
                f"energy_{self.calculator_type}", 0
            )
            if chemical_potential == 0:
                self.log(
                    f"Warning: No chemical potential found for {element} with calculator {self.calculator_type}"
                )
            total_energy -= chemical_potential * amount

        formation_energy_per_atom = total_energy / relaxed_atoms.num_atoms

        # Log and save the formation energy
        self.job_info["formation_energy_per_atom"] = formation_energy_per_atom
        self.log(f"Formation energy per atom: {formation_energy_per_atom}")
        save_dict_to_json(self.job_info, self.get_job_info_filename())

        return formation_energy_per_atom

    def calculate_ev_curve(self, relaxed_atoms):
        """Calculate the energy-volume (E-V) curve and log results."""
        self.log(f"Calculating EV curve for {self.jid}")

        dx = np.arange(-0.06, 0.06, 0.01)  # Strain values
        y = []  # Energies
        vol = []  # Volumes
        strained_structures = []  # To store strained structures

        for i in dx:
            # Apply strain and calculate energy at each strain value
            strained_atoms = relaxed_atoms.strain_atoms(i)
            strained_structures.append(strained_atoms)
            ase_atoms = strained_atoms.ase_converter()
            ase_atoms.calc = self.calculator
            energy = ase_atoms.get_potential_energy()

            y.append(energy)
            vol.append(strained_atoms.volume)

        # Convert data to numpy arrays for processing
        y = np.array(y)
        vol = np.array(vol)

        # Fit the E-V curve using an equation of state (EOS)
        eos = EquationOfState(vol, y, eos="murnaghan")
        v0, e0, B = eos.fit()

        # Bulk modulus in GPa (conversion factor from ASE units)
        kv = B / kJ * 1.0e24  # Convert to GPa

        # Log important results
        self.log(f"Bulk modulus: {kv} GPa")
        self.log(f"Equilibrium energy: {e0} eV")
        self.log(f"Equilibrium volume: {v0} Å³")

        # Save E-V curve plot
        fig = plt.figure()
        eos.plot()
        ev_plot_filename = os.path.join(self.output_dir, "E_vs_V_curve.png")
        fig.savefig(ev_plot_filename)
        plt.close(fig)
        self.log(f"E-V curve plot saved to {ev_plot_filename}")

        # Save E-V curve data to a text file
        ev_data_filename = os.path.join(self.output_dir, "E_vs_V_data.txt")
        with open(ev_data_filename, "w") as f:
            f.write("Volume (Å³)\tEnergy (eV)\n")
            for v, e in zip(vol, y):
                f.write(f"{v}\t{e}\n")
        self.log(f"E-V curve data saved to {ev_data_filename}")

        # Update job info with the results
        self.job_info["bulk_modulus"] = kv
        self.job_info["equilibrium_energy"] = e0
        self.job_info["equilibrium_volume"] = v0
        save_dict_to_json(self.job_info, self.get_job_info_filename())

        # Return additional values for thermal expansion analysis
        return vol, y, strained_structures, eos, kv, e0, v0

    def calculate_elastic_tensor(self, relaxed_atoms):
        """
        Calculate the elastic tensor for the relaxed structure using the provided calculator.
        """
        self.log(f"Starting elastic tensor calculation for {self.jid}")
        start_time = time.time()

        # Convert atoms to ASE format and assign the calculator
        ase_atoms = relaxed_atoms.ase_converter()
        ase_atoms.calc = self.calculator

        # Get elementary deformations for elastic tensor calculation
        systems = elastic.get_elementary_deformations(ase_atoms)

        # Calculate the elastic tensor and convert to GPa
        cij_order = elastic.elastic.get_cij_order(ase_atoms)
        Cij, Bij = elastic.get_elastic_tensor(ase_atoms, systems)
        elastic_tensor = {
            i: j / ase.units.GPa for i, j in zip(cij_order, Cij)
        }  # Convert to GPa

        # Save and log the results
        self.job_info["elastic_tensor"] = elastic_tensor
        self.log(
            f"Elastic tensor for {self.jid} with {self.calculator_type}: {elastic_tensor}"
        )

        # Timing the calculation
        end_time = time.time()
        self.log(f"Elastic Calculation time: {end_time - start_time} seconds")

        # Save to job info JSON
        save_dict_to_json(self.job_info, self.get_job_info_filename())

        return elastic_tensor

    def run_phonon_analysis(self, relaxed_atoms):
        """Perform Phonon calculation, generate force constants, and plot band structure & DOS."""
        self.log(f"Starting phonon analysis for {self.jid}")
        phonopy_bands_figname = f"ph_{self.jid}_{self.calculator_type}.png"

        # Phonon generation parameters
        dim = [2, 2, 2]
        freq_conversion_factor = 33.3566830  # THz to cm^-1
        force_constants_filename = "FORCE_CONSTANTS"
        eigenvalues_filename = "phonon_eigenvalues.txt"
        thermal_props_filename = "thermal_properties.txt"
        write_fc = True
        min_freq_tol = -0.05
        distance = 0.2

        # Generate k-point path
        kpoints = Kpoints().kpath(relaxed_atoms, line_density=5)

        # Convert atoms to Phonopy-compatible object
        self.log("Converting atoms to Phonopy-compatible format...")
        bulk = relaxed_atoms.phonopy_converter()
        phonon = Phonopy(bulk, [[dim[0], 0, 0], [0, dim[1], 0], [0, 0, dim[2]]])

        # Generate displacements
        phonon.generate_displacements(distance=distance)
        supercells = phonon.supercells_with_displacements
        self.log(f"Generated {len(supercells)} supercells for displacements.")

        # Calculate forces for each supercell
        set_of_forces = []
        for idx, scell in enumerate(supercells):
            self.log(f"Calculating forces for supercell {idx+1}...")
            ase_atoms = AseAtoms(
                symbols=scell.symbols,
                positions=scell.positions,
                cell=scell.cell,
                pbc=True,
            )
            ase_atoms.calc = self.calculator
            forces = np.array(ase_atoms.get_forces())

            # Correct for drift force
            drift_force = forces.sum(axis=0)
            for force in forces:
                force -= drift_force / forces.shape[0]

            set_of_forces.append(forces)

        # Generate force constants
        self.log("Producing force constants...")
        phonon.produce_force_constants(forces=set_of_forces)

        # Write force constants to file if required
        if write_fc:
            force_constants_filepath = os.path.join(
                self.output_dir, force_constants_filename
            )
            self.log(f"Writing force constants to {force_constants_filepath}...")
            write_FORCE_CONSTANTS(
                phonon.force_constants, filename=force_constants_filepath
            )
            self.log(f"Force constants saved to {force_constants_filepath}")

        # Phonon band structure and eigenvalues
        lbls = kpoints.labels
        lbls_ticks = []
        freqs = []
        lbls_x = []
        count = 0
        for ii, k in enumerate(kpoints.kpts):
            k_str = ",".join(map(str, k))
            if ii == 0 or k_str != ",".join(map(str, kpoints.kpts[ii - 1])):
                freqs_at_k = phonon.get_frequencies(k)
                freqs.append(freqs_at_k)
                lbl = "$" + str(lbls[ii]) + "$" if lbls[ii] else ""
                if lbl:
                    lbls_ticks.append(lbl)
                    lbls_x.append(count)
                count += 1
        eigenvalues = []

        for k in kpoints.kpts:
            freqs_at_k = phonon.get_frequencies(k)
            freqs.append(freqs_at_k)
            eigenvalues.append((k, freqs_at_k))

        # Write eigenvalues to file
        eigenvalues_filepath = os.path.join(self.output_dir, eigenvalues_filename)
        self.log(f"Writing phonon eigenvalues to {eigenvalues_filepath}...")
        with open(eigenvalues_filepath, "w") as eig_file:
            eig_file.write("k-points\tFrequencies (cm^-1)\n")
            for k, freqs_at_k in eigenvalues:
                k_str = ",".join(map(str, k))
                freqs_str = "\t".join(map(str, freqs_at_k * freq_conversion_factor))
                eig_file.write(f"{k_str}\t{freqs_str}\n")
        self.log(f"Phonon eigenvalues saved to {eigenvalues_filepath}")

        # Convert frequencies to numpy and apply conversion factor
        freqs = np.array(freqs) * freq_conversion_factor

        # Plot phonon band structure and DOS
        the_grid = plt.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.0)
        plt.rcParams.update({"font.size": 18})
        plt.figure(figsize=(10, 5))

        # Plot phonon bands
        plt.subplot(the_grid[0])
        for i in range(freqs.shape[1]):
            plt.plot(freqs[:, i], lw=2, c="b")
        for i in lbls_x:
            plt.axvline(x=i, c="black")
        plt.xticks(lbls_x, lbls_ticks)
        plt.ylabel("Frequency (cm$^{-1}$)")
        plt.xlim([0, max(lbls_x)])

        # Run mesh and DOS calculations
        phonon.run_mesh([40, 40, 40], is_gamma_center=True, is_mesh_symmetry=False)
        phonon.run_total_dos()
        tdos = phonon.total_dos
        freqs_dos = np.array(tdos.frequency_points) * freq_conversion_factor
        dos_values = tdos.dos
        min_freq = min_freq_tol * freq_conversion_factor
        max_freq = max(freqs_dos)

        plt.ylim([min_freq, max_freq])

        # Plot DOS
        plt.subplot(the_grid[1])
        plt.fill_between(
            dos_values, freqs_dos, color=(0.2, 0.4, 0.6, 0.6), edgecolor="k", lw=1, y2=0
        )
        plt.xlabel("DOS")
        plt.yticks([])
        plt.xticks([])
        plt.ylim([min_freq, max_freq])
        plt.xlim([0, max(dos_values)])

        # Save the plot
        os.makedirs(self.output_dir, exist_ok=True)
        plot_filepath = os.path.join(self.output_dir, phonopy_bands_figname)
        plt.tight_layout()
        plt.savefig(plot_filepath)
        self.log(
            f"Phonon band structure and DOS combined plot saved to {plot_filepath}"
        )
        plt.close()

        self.log("Calculating thermal properties...")
        phonon.run_mesh(mesh=[20, 20, 20])
        phonon.run_thermal_properties(t_step=10, t_max=1000, t_min=0)
        tprop_dict = phonon.get_thermal_properties_dict()

        # Plot thermal properties
        plt.figure()
        plt.plot(
            tprop_dict["temperatures"],
            tprop_dict["free_energy"],
            label="Free energy (kJ/mol)",
            color="red",
        )
        plt.plot(
            tprop_dict["temperatures"],
            tprop_dict["entropy"],
            label="Entropy (J/K*mol)",
            color="blue",
        )
        plt.plot(
            tprop_dict["temperatures"],
            tprop_dict["heat_capacity"],
            label="Heat capacity (J/K*mol)",
            color="green",
        )
        plt.legend()
        plt.xlabel("Temperature (K)")
        plt.ylabel("Thermal Properties")
        plt.title("Thermal Properties")

        thermal_props_plot_filepath = os.path.join(
            self.output_dir, f"Thermal_Properties_{self.jid}.png"
        )
        plt.savefig(thermal_props_plot_filepath)
        self.log(f"Thermal properties plot saved to {thermal_props_plot_filepath}")
        plt.close()

        # Save thermal properties to file
        thermal_props_filepath = os.path.join(self.output_dir, thermal_props_filename)
        with open(thermal_props_filepath, "w") as f:
            f.write(
                "Temperature (K)\tFree Energy (kJ/mol)\tEntropy (J/K*mol)\tHeat Capacity (J/K*mol)\n"
            )
            for i in range(len(tprop_dict["temperatures"])):
                f.write(
                    f"{tprop_dict['temperatures'][i]}\t{tprop_dict['free_energy'][i]}\t"
                    f"{tprop_dict['entropy'][i]}\t{tprop_dict['heat_capacity'][i]}\n"
                )
        self.log(f"Thermal properties written to {thermal_props_filepath}")

        # Calculate zero-point energy (ZPE)
        zpe = tprop_dict["free_energy"][0] * 0.0103643  # Converting from kJ/mol to eV
        self.log(f"Zero-point energy: {zpe} eV")

        # Save to job info
        self.job_info["phonopy_bands"] = phonopy_bands_figname
        save_dict_to_json(self.job_info, self.get_job_info_filename())

        return phonon, zpe

    def analyze_defects(self):
        """Analyze defects by generating, relaxing, and calculating vacancy formation energy."""
        self.log("Starting defect analysis...")

    # Generate defect structures from the original atoms
        defect_structures = Vacancy(self.atoms).generate_defects(on_conventional_cell=True, enforce_c_size=8, extend=1)

        for defect in defect_structures:
        # Extract the defect structure and related metadata
            defect_structure = Atoms.from_dict(defect.to_dict()["defect_structure"])
        
        # Construct a consistent defect name without Wyckoff notation
            element = defect.to_dict()['symbol']
            defect_name = f"{self.jid}_{element}"  # Consistent format
            self.log(f"Analyzing defect: {defect_name}")

        # Relax the defect structure
            relaxed_defect_atoms = self.relax_defect_structure(defect_structure, name=defect_name)

            if relaxed_defect_atoms is None:
                self.log(f"Skipping {defect_name} due to failed relaxation.")
                continue

        # Retrieve energies for calculating the vacancy formation energy
            vacancy_energy = self.job_info.get(f"final_energy_defect for {defect_name}")
            bulk_energy = (
                self.job_info.get("equilibrium_energy")
                / self.atoms.num_atoms
                * (defect_structure.num_atoms + 1)
            )

            if vacancy_energy is None or bulk_energy is None:
                self.log(f"Skipping {defect_name} due to missing energy values.")
                continue

        # Get chemical potential and calculate vacancy formation energy
            chemical_potential = self.get_chemical_potential(element)

            if chemical_potential is None:
                self.log(f"Skipping {defect_name} due to missing chemical potential for {element}.")
                continue

            vacancy_formation_energy = vacancy_energy - bulk_energy + chemical_potential

        # Log and store the vacancy formation energy consistently
            self.job_info[f"vacancy_formation_energy for {defect_name}"] = vacancy_formation_energy
            self.log(f"Vacancy formation energy for {defect_name}: {vacancy_formation_energy} eV")

    # Save the job info to a JSON file
        save_dict_to_json(self.job_info, self.get_job_info_filename())
        self.log("Defect analysis completed.")


    def relax_defect_structure(self, atoms, name):
        """Relax the defect structure and log the process."""
        # Convert atoms to ASE format and assign the calculator
        ase_atoms = atoms.ase_converter()
        ase_atoms.calc = self.calculator
        ase_atoms = ExpCellFilter(ase_atoms, constant_volume=True)

        # Run FIRE optimizer and capture the output
        final_energy, nsteps = self.capture_fire_output(ase_atoms, fmax=0.05, steps=200)
        relaxed_atoms = ase_to_atoms(ase_atoms.atoms)
        converged = nsteps < 200

        # Log the final energy and relaxation status
        self.log(
            f"Final energy of FIRE optimization for defect structure: {final_energy}"
        )
        self.log(
            f"Defect relaxation {'converged' if converged else 'did not converge'} within 200 steps."
        )

        # Update job info with the final energy and convergence status
        self.job_info[f"final_energy_defect for {name}"] = final_energy
        self.job_info[f"converged for {name}"] = converged

        return relaxed_atoms if converged else None

    def analyze_surfaces(
            self,
            indices_list=[[1, 0, 0], [1, 1, 1], [1, 1, 0], [0, 1, 1], [0, 0, 1], [0, 1, 0]],
        ):
        """
        Perform surface analysis by generating surface structures, relaxing them, and calculating surface energies.
        """
        self.log(f"Analyzing surfaces for {self.jid}")

        for indices in indices_list:
            # Generate surface and check for polarity
            surface = (
                Surface(atoms=self.atoms, indices=indices, layers=4, vacuum=18)
                .make_surface()
                .center_around_origin()
            )
            if surface.check_polar:
                self.log(
                    f"Skipping polar surface for {self.jid} with indices {indices}"
                )
                continue

            # Write initial POSCAR for surface
            poscar_surface = Poscar(atoms=surface)
            poscar_surface.write_file(
                os.path.join(
                    self.output_dir,
                    f"POSCAR_{self.jid}_surface_{indices}_{self.calculator_type}.vasp",
                )
            )

            # Relax the surface structure
            relaxed_surface_atoms, final_energy = self.relax_surface_structure(
                surface, indices
            )

            # If relaxation failed, skip further calculations
            if relaxed_surface_atoms is None:
                self.log(f"Skipping surface {indices} due to failed relaxation.")
                continue

            # Write relaxed POSCAR for surface
            pos_relaxed_surface = Poscar(relaxed_surface_atoms)
            pos_relaxed_surface.write_file(
                os.path.join(
                    self.output_dir,
                    f"POSCAR_{self.jid}_surface_{indices}_{self.calculator_type}_relaxed.vasp",
                )
            )

            # Calculate and log surface energy
            bulk_energy = self.job_info.get("equilibrium_energy")
            if final_energy is None or bulk_energy is None:
                self.log(
                    f"Skipping surface energy calculation for {self.jid} with indices {indices} due to missing energy values."
                )
                continue

            surface_energy = self.calculate_surface_energy(
                final_energy, bulk_energy, relaxed_surface_atoms, surface
            )

            # Store the surface energy with the new naming convention
            surface_name = f"Surface-{self.jid}_miller_{'_'.join(map(str, indices))}"
            self.job_info[surface_name] = surface_energy
            self.log(
                f"Surface energy for {self.jid} with indices {indices}: {surface_energy} J/m^2"
            )

        # Save updated job info
        save_dict_to_json(
            self.job_info,
            os.path.join(
                self.output_dir, f"{self.jid}_{self.calculator_type}_job_info.json"
            ),
        )
        self.log("Surface analysis completed.")


    def relax_surface_structure(self, atoms, indices):
        """
        Relax the surface structure and log the process.
        """
        self.log(f"Starting surface relaxation for {self.jid} with indices {indices}")
        start_time = time.time()

        # Convert atoms to ASE format and assign the calculator
        ase_atoms = atoms.ase_converter()
        ase_atoms.calc = self.calculator
        ase_atoms = ExpCellFilter(ase_atoms, constant_volume=True)

        # Run FIRE optimizer and capture the output
        final_energy, nsteps = self.capture_fire_output(ase_atoms, fmax=0.05, steps=200)
        relaxed_atoms = ase_to_atoms(ase_atoms.atoms)
        converged = nsteps < 200

        # Log relaxation results
        self.log(
            f"Final energy of FIRE optimization for surface structure: {final_energy}"
        )
        self.log(
            f"Surface relaxation {'converged' if converged else 'did not converge'} within {nsteps} steps."
        )

        end_time = time.time()
        self.log(
            f"Surface Relaxation Calculation time: {end_time - start_time} seconds"
        )

        # Update job info and return relaxed atoms if converged, otherwise return None
        self.job_info[f"final_energy_surface_{indices}"] = final_energy
        self.job_info[f"converged_surface_{indices}"] = converged

        # Return both relaxed atoms and the final energy as a tuple
        return (relaxed_atoms if converged else None), final_energy

    def calculate_surface_energy(
        self, final_energy, bulk_energy, relaxed_atoms, surface
    ):
        """
        Calculate the surface energy based on the final energy of the relaxed surface and bulk energy.
        """
        # Calculate the number of bulk units in the surface supercell
        num_units = surface.num_atoms / self.atoms.num_atoms

        # Calculate the surface area using the lattice vectors
        m = relaxed_atoms.lattice.matrix
        surface_area = np.linalg.norm(np.cross(m[0], m[1]))

        # Calculate surface energy in J/m^2
        surface_energy = (
            (final_energy - bulk_energy * num_units) * 16.02176565 / (2 * surface_area)
        )

        return surface_energy

    def run_phonon3_analysis(self, relaxed_atoms):
        """Run Phono3py analysis, process results, and generate thermal conductivity data."""
        self.log(f"Starting Phono3py analysis for {self.jid}")

        # Set parameters for the Phono3py calculation
        dim = [2, 2, 2]
        distance = 0.2
        force_multiplier = 16

        # Convert atoms to Phonopy-compatible object and set up Phono3py
        ase_atoms = relaxed_atoms.ase_converter()
        ase_atoms.calc = self.calculator
        bulk = relaxed_atoms.phonopy_converter()

        phonon = Phono3py(bulk, [[dim[0], 0, 0], [0, dim[1], 0], [0, 0, dim[2]]])
        phonon.generate_displacements(distance=distance)
        supercells = phonon.supercells_with_displacements

        # Calculate forces for each supercell
        set_of_forces = []
        for scell in supercells:
            ase_atoms = AseAtoms(
                symbols=scell.get_chemical_symbols(),
                scaled_positions=scell.get_scaled_positions(),
                cell=scell.get_cell(),
                pbc=True,
            )
            ase_atoms.calc = self.calculator
            forces = np.array(ase_atoms.get_forces())
            drift_force = forces.sum(axis=0)
            for force in forces:
                force -= drift_force / forces.shape[0]
            set_of_forces.append(forces)

        # Set the forces and produce third-order force constants
        forces = np.array(set_of_forces).reshape(-1, len(phonon.supercell), 3)
        phonon.forces = forces
        phonon.produce_fc3()

        # Run thermal conductivity calculation
        phonon.mesh_numbers = 30
        phonon.init_phph_interaction()
        phonon.run_thermal_conductivity(
            temperatures=range(0, 1001, 10), write_kappa=True
        )

        # Collect thermal conductivity data
        kappa = phonon.thermal_conductivity.kappa
        self.log(f"Thermal conductivity: {kappa}")

        # Move generated HDF5 files to the output directory
        hdf5_file_pattern = "kappa-*.hdf5"
        for hdf5_file in glob.glob(hdf5_file_pattern):
            shutil.move(hdf5_file, os.path.join(self.output_dir, hdf5_file))

        # Process Phono3py results and save plots
        self.process_phonon3_results()

        # Save updated job info to JSON
        save_dict_to_json(
            self.job_info,
            os.path.join(
                self.output_dir, f"{self.jid}_{self.calculator_type}_job_info.json"
            ),
        )
        self.log(f"Phono3py analysis completed for {self.jid}")

    def process_phonon3_results(self):
        """Process Phono3py results and generate plots of thermal conductivity."""
        file_pattern = os.path.join(self.output_dir, "kappa-*.hdf5")
        file_list = glob.glob(file_pattern)

        temperatures = np.arange(10, 101, 10)
        kappa_xx_values = []

        if file_list:
            hdf5_filename = file_list[0]
            self.log(f"Processing file: {hdf5_filename}")

            for temperature_index in temperatures:
                converted_kappa = self.convert_kappa_units(
                    hdf5_filename, temperature_index
                )
                kappa_xx = converted_kappa[0]
                kappa_xx_values.append(kappa_xx)
                self.log(
                    f"Temperature index {temperature_index}, converted kappa: {kappa_xx}"
                )

            # Save results to job_info
            self.job_info["temperatures"] = temperatures.tolist()
            self.job_info["kappa_xx_values"] = kappa_xx_values

            # Plot temperature vs. converted kappa (xx element)
            plt.figure(figsize=(8, 6))
            plt.plot(
                temperatures * 10, kappa_xx_values, marker="o", linestyle="-", color="b"
            )
            plt.xlabel("Temperature (K)")
            plt.ylabel("Converted Kappa (xx element)")
            plt.title("Temperature vs. Converted Kappa (xx element)")
            plt.grid(True)
            plt.savefig(
                os.path.join(self.output_dir, "Temperature_vs_Converted_Kappa.png")
            )
            plt.close()
        else:
            self.log("No files matching the pattern were found.")

    def convert_kappa_units(self, hdf5_filename, temperature_index):
        """Convert thermal conductivity kappa from HDF5 file units."""
        with h5py.File(hdf5_filename, "r") as f:
            kappa_unit_conversion = f["kappa_unit_conversion"][()]
            heat_capacity = f["heat_capacity"][:]
            gv_by_gv = f["gv_by_gv"][:]
            gamma = f["gamma"][:]

            converted_kappa = (
                kappa_unit_conversion
                * heat_capacity[temperature_index, 2, 0]
                * gv_by_gv[2, 0]
                / (2 * gamma[temperature_index, 2, 0])
            )

            return converted_kappa

    def calculate_thermal_expansion(self, relaxed_atoms):
        """Calculate the thermal expansion coefficient using QHA."""

        def log(message):
            with open(self.log_file, "a") as f:
                f.write(message + "\n")
            print(message)

        log("Starting thermal expansion analysis...")

        # Step 1: Calculate finer E-V curve
        vol, y, strained_structures, eos, kv, e0, v0 = self.fine_ev_curve(
            atoms=relaxed_atoms, dx=np.linspace(-0.05, 0.05, 50)  # Denser grid
        )

        # Log Bulk modulus, equilibrium energy, and volume
        log(
            f"Bulk modulus: {kv} GPa, Equilibrium energy: {y[0]} eV, Volume: {vol[0]} Å³"
        )
        self.job_info["bulk_modulus"] = kv
        self.job_info["equilibrium_energy"] = y[0]
        self.job_info["equilibrium_volume"] = vol[0]

        # Step 2: Generate phonons for strained structures
        free_energies, heat_capacities, entropies, temperatures = (
            self.generate_phonons_for_volumes(
                strained_structures,
                calculator=self.calculator,
                dim=[2, 2, 2],
                distance=0.2,
                mesh=[20, 20, 20],
            )
        )

        # Step 3: Perform QHA-based thermal expansion analysis
        alpha = self.perform_qha(
            volumes=vol,
            energies=y,
            free_energies=free_energies,
            heat_capacities=heat_capacities,
            entropies=entropies,
            temperatures=temperatures,
            output_dir=self.output_dir,
        )

        self.log(f"Thermal expansion coefficient calculated: {alpha}")
        save_dict_to_json(
            self.job_info,
            os.path.join(
                self.output_dir, f"{self.jid}_{self.calculator_type}_job_info.json"
            ),
        )
        self.log(
            f"Thermal expansion analysis information saved to file: {self.jid}_{self.calculator_type}_job_info.json"
        )

    # Helper Functions Inside the Class
    def fine_ev_curve(self, atoms, dx=np.linspace(-0.05, 0.05, 50)):
        """
        Generate a finer energy-volume curve for strained structures.
        """
        y = []
        vol = []
        strained_structures = []

        for i in dx:
            # Apply strain and get strained atoms
            strained_atoms = atoms.strain_atoms(i)
            ase_atoms = strained_atoms.ase_converter()  # Convert to ASE Atoms
            ase_atoms.calc = self.calculator  # Assign the calculator

            # Get potential energy and volume
            energy = ase_atoms.get_potential_energy()
            y.append(energy)
            vol.append(strained_atoms.volume)

            strained_structures.append(strained_atoms)  # Save the strained structure

        vol = np.array(vol)
        y = np.array(y)

        # Fit the E-V curve using an equation of state (EOS)
        eos = EquationOfState(vol, y, eos="murnaghan")
        v0, e0, B = eos.fit()
        kv = B / kJ * 1.0e24  # Convert to GPa

        # Log important results
        self.log(f"Bulk modulus: {kv} GPa")
        self.log(f"Equilibrium energy: {e0} eV")
        self.log(f"Equilibrium volume: {v0} Å³")

        # Save E-V curve plot
        fig = plt.figure()
        eos.plot()
        ev_plot_filename = os.path.join(self.output_dir, "E_vs_V_curve.png")
        fig.savefig(ev_plot_filename)
        plt.close(fig)
        self.log(f"E-V curve plot saved to {ev_plot_filename}")

        # Save E-V curve data to a text file
        ev_data_filename = os.path.join(self.output_dir, "E_vs_V_data.txt")
        with open(ev_data_filename, "w") as f:
            f.write("Volume (Å³)\tEnergy (eV)\n")
            for v, e in zip(vol, y):
                f.write(f"{v}\t{e}\n")
        self.log(f"E-V curve data saved to {ev_data_filename}")

        # Update job info with the results
        self.job_info["bulk_modulus"] = kv
        self.job_info["equilibrium_energy"] = e0
        self.job_info["equilibrium_volume"] = v0
        save_dict_to_json(self.job_info, self.get_job_info_filename())

        return vol, y, strained_structures, eos, kv, e0, v0

    def generate_phonons_for_volumes(
        self, structures, calculator, dim=[2, 2, 2], distance=0.2, mesh=[20, 20, 20]
    ):
        all_free_energies = []
        all_heat_capacities = []
        all_entropies = []
        temperatures = np.arange(0, 300, 6)  # Define temperature range

        for structure in structures:
            # Convert structure to PhonopyAtoms
            phonopy_atoms = PhonopyAtoms(
                symbols=[str(e) for e in structure.elements],
                positions=structure.cart_coords,
                cell=structure.lattice.matrix,
            )

            # Initialize Phonopy object
            phonon = Phonopy(
                phonopy_atoms, [[dim[0], 0, 0], [0, dim[1], 0], [0, 0, dim[2]]]
            )
            phonon.generate_displacements(distance=distance)

            # Calculate forces for displaced structures
            supercells = phonon.get_supercells_with_displacements()
            forces = []
            for scell in supercells:
                ase_atoms = AseAtoms(
                    symbols=scell.symbols,
                    positions=scell.positions,
                    cell=scell.cell,
                    pbc=True,
                )
                ase_atoms.calc = calculator
                forces.append(ase_atoms.get_forces())

            phonon.produce_force_constants(forces=forces)

            # Post-processing to get thermal properties
            phonon.run_mesh(mesh=mesh)
            phonon.run_thermal_properties(t_min=0, t_step=6, t_max=294)
            tprop_dict = phonon.get_thermal_properties_dict()

            free_energies = tprop_dict["free_energy"]
            heat_capacities = tprop_dict["heat_capacity"]
            entropies = tprop_dict["entropy"]

            all_entropies.append(entropies)
            all_free_energies.append(free_energies)
            all_heat_capacities.append(heat_capacities)

        return (
            np.array(all_free_energies),
            np.array(all_heat_capacities),
            np.array(all_entropies),
            temperatures,
        )

    def perform_qha(
        self,
        volumes,
        energies,
        free_energies,
        heat_capacities,
        entropies,
        temperatures,
        output_dir,
    ):
        # Debugging: print array sizes
        print(f"Number of temperatures: {len(temperatures)}")
        print(f"Number of free energy data points: {free_energies.shape}")
        print(f"Number of volume data points: {len(volumes)}")

        # Ensure that volumes, free_energies, and temperatures are consistent
        if len(volumes) != len(temperatures):
            raise ValueError(
                "The number of volumes must match the number of temperatures"
            )

        # Initialize the QHA object
        try:
            qha = PhonopyQHA(
                volumes=volumes,
                electronic_energies=energies,
                free_energy=free_energies,
                cv=heat_capacities,
                entropy=entropies,
                temperatures=temperatures,
                eos="murnaghan",  # or another EOS if needed
                verbose=True,
            )
        except IndexError as e:
            print(f"Error in QHA initialization: {e}")
            raise

        # Calculate thermal expansion and save plots
        thermal_expansion_plot = os.path.join(output_dir, "thermal_expansion.png")
        volume_temperature_plot = os.path.join(output_dir, "volume_temperature.png")
        helmholtz_volume_plot = os.path.join(output_dir, "helmholtz_volume.png")

        qha.get_thermal_expansion()

        # Save thermal expansion plot
        qha.plot_thermal_expansion()
        plt.savefig(thermal_expansion_plot)

        # Save volume-temperature plot
        qha.plot_volume_temperature()
        plt.savefig(volume_temperature_plot)

        # Save Helmholtz free energy vs. volume plot
        qha.plot_helmholtz_volume()
        plt.savefig(helmholtz_volume_plot)

        # Optionally save thermal expansion coefficient to a file
        thermal_expansion_file = os.path.join(output_dir, "thermal_expansion.txt")
        alpha = qha.write_thermal_expansion(filename=thermal_expansion_file)

        return alpha

    def general_melter(self, relaxed_atoms):
        """Perform MD simulation to melt the structure, then quench it back to room temperature."""
        self.log(f"Starting MD melting and quenching simulation for {self.jid}")

        calculator = self.setup_calculator()
        ase_atoms = relaxed_atoms.ase_converter()
        dim = self.ensure_cell_size(ase_atoms, min_size=10.0)
        supercell = relaxed_atoms.make_supercell_matrix(dim)
        ase_atoms = supercell.ase_converter()
        ase_atoms.calc = calculator

        dt = 1 * ase.units.fs
        temp0, nsteps0 = 3500, 10
        temp1, nsteps1 = 300, 20
        taut = 20 * ase.units.fs
        trj = os.path.join(self.output_dir, f"{self.jid}_melt.traj")

        # Initialize velocities and run the first part of the MD simulation
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
        from ase.md.nvtberendsen import NVTBerendsen

        MaxwellBoltzmannDistribution(ase_atoms, temp0 * ase.units.kB)
        dyn = NVTBerendsen(ase_atoms, dt, temp0, taut=taut, trajectory=trj)

        def myprint():
            message = f"time={dyn.get_time() / ase.units.fs: 5.0f} fs T={ase_atoms.get_temperature(): 3.0f} K"
            self.log(message)

        dyn.attach(myprint, interval=20)
        dyn.run(nsteps0)

        # Cool down to room temperature
        dyn.set_temperature(temp1)
        dyn.run(nsteps1)

        # Convert back to JARVIS atoms and save the final structure
        final_atoms = ase_to_atoms(ase_atoms)
        poscar_filename = os.path.join(
            self.output_dir, f"POSCAR_{self.jid}_quenched_{self.calculator_type}.vasp"
        )
        from ase.io import write

        write(poscar_filename, final_atoms.ase_converter(), format="vasp")
        self.log(f"MD simulation completed. Final structure saved to {poscar_filename}")
        self.job_info["quenched_atoms"] = final_atoms.to_dict()

        return final_atoms

    def calculate_rdf(self, quenched_atoms):
        """Calculate Radial Distribution Function (RDF) for quenched structure and save plot."""
        self.log(f"Starting RDF calculation for {self.jid}")
        ase_atoms = quenched_atoms.ase_converter()
        rmax = 3.5
        nbins = 200

        def perform_rdf_calculation(rmax):
            from ase.ga.utilities import get_rdf

            rdfs, distances = get_rdf(ase_atoms, rmax, nbins)
            plt.figure()
            plt.plot(distances, rdfs)
            plt.xlabel("Distance (Å)")
            plt.ylabel("RDF")
            plt.title(
                f"Radial Distribution Function for {self.jid} ({self.calculator_type})"
            )
            rdf_plot_filename = os.path.join(
                self.output_dir, f"RDF_{self.jid}_{self.calculator_type}.png"
            )
            plt.savefig(rdf_plot_filename)
            plt.close()
            self.job_info["rdf_plot"] = rdf_plot_filename
            self.log(f"RDF plot saved to {rdf_plot_filename}")
            return rdf_plot_filename

        try:
            perform_rdf_calculation(rmax)
        except ValueError as e:
            if "The cell is not large enough" in str(e):
                recommended_rmax = float(str(e).split("<")[1].split("=")[1])
                self.log(f"Error: {e}. Adjusting rmax to {recommended_rmax}.")
                perform_rdf_calculation(recommended_rmax)
            else:
                self.log(f"Error: {e}")
                raise

    def ensure_cell_size(self, ase_atoms, min_size):
        """Ensure that all cell dimensions are at least min_size."""
        cell_lengths = ase_atoms.get_cell().lengths()
        scale_factors = np.ceil(min_size / cell_lengths).astype(int)
        supercell_dims = [max(1, scale) for scale in scale_factors]
        return supercell_dims

    def get_chemical_potential(self, element):
        """Fetch chemical potential from JSON based on the element and calculator."""
        if element in self.chemical_potentials:
            return self.chemical_potentials[element].get(
                f"energy_{self.calculator_type}", 0
            )
        else:
            self.log(
                f"Warning: No chemical potential data found for element: {element}"
            )
            return 0

    def get_job_info_filename(self):
        return os.path.join(
            self.output_dir, f"{self.jid}_{self.calculator_type}_job_info.json"
        )

    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, r2_score
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import os

    def run_all(self):
        """Run all analyses in sequence and output results in the required JSON format."""
        # Initialize lists to store errors
        timings = []
        error_dat = {}

        # Start timing the entire run
        start_time = time.time()

        # Relax the structure
        relaxed_atoms = self.relax_structure()

        # Lattice parameters before and after relaxation
        lattice_initial = self.atoms.lattice
        lattice_final = relaxed_atoms.lattice

        # Bulk modulus (E-V curve)
        _, _, _, _, bulk_modulus, _, _ = self.calculate_ev_curve(relaxed_atoms)
        kv_entry = self.reference_data.get("bulk_modulus_kv", 0)

        # Formation energy
        formation_energy = self.calculate_formation_energy(relaxed_atoms)
        form_en_entry = self.reference_data.get("formation_energy_peratom", 0)

        # Elastic tensor
        elastic_tensor = self.calculate_elastic_tensor(relaxed_atoms)
        c11_entry = self.reference_data.get("elastic_tensor", [[0]])[0][0]
        c44_entry = self.reference_data.get("elastic_tensor", [[0, 0, 0, [0, 0, 0, 0]]])[3][3]

        # Surface energy analysis
        self.analyze_surfaces()
        surf_en, surf_en_entry = [], []
        surface_entries = get_surface_energy_entry(self.jid, collect_data(dft_3d, vacancydb, surface_data))

        # Handle surface energies and skip 0 values
        for indices in [[1, 0, 0], [1, 1, 1], [1, 1, 0], [0, 1, 1], [0, 0, 1], [0, 1, 0]]:
            surface_name = f"Surface-{self.jid}_miller_{'_'.join(map(str, indices))}"
            calculated_surface_energy = self.job_info.get(surface_name, 0)

            matching_entry = next((entry for entry in surface_entries if entry['name'].strip() == surface_name.strip()), None)

            if matching_entry and calculated_surface_energy != 0 and matching_entry["surf_en_entry"] != 0:
                surf_en.append(calculated_surface_energy)
                surf_en_entry.append(matching_entry["surf_en_entry"])

        # Vacancy energy analysis
        self.analyze_defects()
        vac_en, vac_en_entry = [], []
        vacancy_entries = get_vacancy_energy_entry(self.jid, collect_data(dft_3d, vacancydb, surface_data))

        # Handle vacancy energies and skip 0 values
        for defect in Vacancy(self.atoms).generate_defects(on_conventional_cell=True, enforce_c_size=8, extend=1):
            defect_name = f"{self.jid}_{defect.to_dict()['symbol']}"
            vacancy_energy = self.job_info.get(f"vacancy_formation_energy for {defect_name}", 0)

            matching_entry = next((entry for entry in vacancy_entries if entry['symbol'] == defect_name), None)

            if matching_entry and vacancy_energy != 0 and matching_entry['vac_en_entry'] != 0:
                vac_en.append(vacancy_energy)
                vac_en_entry.append(matching_entry['vac_en_entry'])

        self.run_phonon_analysis(relaxed_atoms)
        self.run_phonon3_analysis(relaxed_atoms)
        self.calculate_thermal_expansion(relaxed_atoms)
        quenched_atoms = self.general_melter(relaxed_atoms)
        
        # Calculate error metrics
        err_a = mean_absolute_error([lattice_initial.a], [lattice_final.a])
        err_b = mean_absolute_error([lattice_initial.b], [lattice_final.b])
        err_c = mean_absolute_error([lattice_initial.c], [lattice_final.c])
        err_form = mean_absolute_error([form_en_entry], [formation_energy])
        err_vol = mean_absolute_error([lattice_initial.volume], [lattice_final.volume])
        err_kv = mean_absolute_error([kv_entry], [bulk_modulus])
        err_c11 = mean_absolute_error([c11_entry], [elastic_tensor.get("C_11", 0)])
        err_c44 = mean_absolute_error([c44_entry], [elastic_tensor.get("C_44", 0)])

        if len(surf_en) > 0:
            err_surf_en = mean_absolute_error(surf_en_entry, surf_en)
        else:
            err_surf_en = 0

        if len(vac_en) > 0:
            err_vac_en = mean_absolute_error(vac_en_entry, vac_en)
        else:
            err_vac_en = 0

        end_time = time.time()
        total_time = end_time - start_time

        # Create an error dictionary
        error_dat["err_a"] = err_a
        error_dat["err_b"] = err_b
        error_dat["err_c"] = err_c
        error_dat["err_form"] = err_form
        error_dat["err_vol"] = err_vol
        error_dat["err_kv"] = err_kv
        error_dat["err_c11"] = err_c11
        error_dat["err_c44"] = err_c44
        error_dat["err_surf_en"] = err_surf_en
        error_dat["err_vac_en"] = err_vac_en
        error_dat["time"] = total_time

        print("Error metrics calculated:", error_dat)

        # Create a DataFrame for error data
        df = pd.DataFrame([error_dat])

        # Save the DataFrame to CSV
        unique_dir = os.path.basename(self.output_dir)
        fname = os.path.join(self.output_dir, f"{unique_dir}_error_dat.csv")
        df.to_csv(fname, index=False)

        # Plot the scorecard with errors
        self.plot_error_scorecard(df)

        return error_dat

    def plot_error_scorecard(self, df):
        import plotly.express as px

        fig = px.imshow(df, text_auto=True, aspect="auto", labels=dict(color="Error"))
        unique_dir = os.path.basename(self.output_dir)
        fname_plot = os.path.join(self.output_dir, f"{unique_dir}_error_scorecard.png")
        fig.write_image(fname_plot)
        fig.show()
        
    def plot_results(self, fname):
        df = pd.read_csv(fname)

        plt.rcParams.update({"font.size": 18})
        plt.figure(figsize=(16, 14))
        the_grid = GridSpec(4, 3)

        # Plot lattice parameter a
        plt.subplot(the_grid[0, 0])
        plt.scatter(df["initial_a"], df["final_a"])
        plt.plot(df["initial_a"], df["initial_a"], c="black", linestyle="-.")
        plt.xlabel("a-DFT ($\AA$)")
        plt.ylabel("a-ML ($\AA$)")
        title = "(a) " + str(round(r2_score(df["initial_a"], df["final_a"]), 2))
        plt.title(title)

        # Plot lattice parameter b
        plt.subplot(the_grid[0, 1])
        plt.scatter(df["initial_b"], df["final_b"])
        plt.plot(df["initial_b"], df["initial_b"], c="black", linestyle="-.")
        plt.xlabel("b-DFT ($\AA$)")
        plt.ylabel("b-ML ($\AA$)")
        title = "(b) " + str(round(r2_score(df["initial_b"], df["final_b"]), 2))
        plt.title(title)

        # Plot lattice parameter c
        plt.subplot(the_grid[0, 2])
        plt.scatter(df["initial_c"], df["final_c"])
        plt.plot(df["initial_c"], df["initial_c"], c="black", linestyle="-.")
        plt.xlabel("c-DFT ($\AA$)")
        plt.ylabel("c-ML ($\AA$)")
        title = "(c) " + str(round(r2_score(df["initial_c"], df["final_c"]), 2))
        plt.title(title)

        # Plot formation energy
        plt.subplot(the_grid[1, 0])
        plt.scatter(df["form_en_entry"], df["form_en"])
        plt.plot(df["form_en_entry"], df["form_en_entry"], c="black", linestyle="-.")
        plt.xlabel("$E_f$-DFT (eV/atom)")
        plt.ylabel("$E_f$-ML (eV/atom)")
        title = "(d) " + str(round(r2_score(df["form_en_entry"], df["form_en"]), 2))
        plt.title(title)

        # Plot volume
        plt.subplot(the_grid[1, 1])
        plt.scatter(df["initial_vol"], df["final_vol"])
        plt.plot(df["initial_vol"], df["initial_vol"], c="black", linestyle="-.")
        plt.xlabel("vol-DFT (${\AA}^3$)")
        plt.ylabel("vol-ML (${\AA}^3$)")
        title = "(e) " + str(round(r2_score(df["initial_vol"], df["final_vol"]), 2))
        plt.title(title)

        # Plot C11
        plt.subplot(the_grid[1, 2])
        plt.scatter(df["c11_entry"], df["c11"])
        plt.plot(df["c11_entry"], df["c11_entry"], c="black", linestyle="-.")
        plt.xlabel("$C_{11}$-DFT (GPa)")
        plt.ylabel("$C_{11}$-ML (GPa)")
        title = "(f) " + str(round(r2_score(df["c11_entry"], df["c11"]), 2))
        plt.title(title)

        # Plot C44
        plt.subplot(the_grid[2, 0])
        plt.scatter(df["c44_entry"], df["c44"])
        plt.plot(df["c44_entry"], df["c44_entry"], c="black", linestyle="-.")
        plt.xlabel("$C_{44}$-DFT (GPa)")
        plt.ylabel("$C_{44}$-ML (GPa)")
        title = "(g) " + str(round(r2_score(df["c44_entry"], df["c44"]), 2))
        plt.title(title)

        # Plot vacancy energy
        vac_x = np.array(df["vac_en_entry"].iloc[0].split(";"), dtype=float)
        vac_y = np.array(df["vac_en"].iloc[0].split(";"), dtype=float)
        plt.subplot(the_grid[2, 1])
        plt.scatter(vac_x, vac_y)
        plt.plot(vac_x, vac_x, linestyle="-.", c="black")
        title = "(h) " + str(round(r2_score(vac_x, vac_y), 2))
        plt.title(title)
        plt.xlabel("$E_{vac}$-DFT (eV)")
        plt.ylabel("$E_{vac}$-ML (eV)")

        # Plot surface energy
        surf_x = np.array(df["surf_en_entry"].iloc[0].split(";"), dtype=float)
        surf_y = np.array(df["surf_en"].iloc[0].split(";"), dtype=float)
        plt.subplot(the_grid[2, 2])
        plt.scatter(surf_x, surf_y)
        plt.plot(surf_x, surf_x, linestyle="-.", c="black")
        title = "(i) " + str(round(r2_score(surf_x, surf_y), 2))
        plt.title(title)
        plt.xlabel("$E_{surf}$-DFT (J/m2)")
        plt.ylabel("$E_{surf}$-ML (J/m2)")

        plt.tight_layout()
        pname = fname.replace("dat.csv", "dat.png")
        plt.savefig(pname)
        plt.close()

def analyze_multiple_structures(jid_list, calculator_types, chemical_potentials_json):
    for jid in jid_list:
        for calculator_type in calculator_types:
            print(f"Analyzing {jid} with {calculator_type}...")
            analyzer = MaterialsAnalyzer(
                jid=jid,
                calculator_type=calculator_type,
                chemical_potentials_json=chemical_potentials_json,
            )
            analyzer.run_all()

def analyze_multiple_structures(jid_list, calculator_types, chemical_potentials_json):
    composite_error_data = {}

    for calculator_type in calculator_types:
        # Initialize accumulators for errors and time
        cumulative_error = {
            "err_a": 0, "err_b": 0, "err_c": 0,
            "err_form": 0, "err_vol": 0,
            "err_c11": 0, "err_c44": 0,
            "err_surf_en": 0, "err_vac_en": 0,
            "time": 0
        }
        num_materials = len(jid_list)  # Number of materials

        for jid in jid_list:
            print(f"Analyzing {jid} with {calculator_type}...")
            analyzer = MaterialsAnalyzer(
                jid=jid,
                calculator_type=calculator_type,
                chemical_potentials_json=chemical_potentials_json,
            )
            # Run analysis and get error data
            error_dat = analyzer.run_all()

            # Aggregate errors
            for key in cumulative_error.keys():
                cumulative_error[key] += error_dat[key]

        # Compute average errors (if you want to average them over materials)
        for key in cumulative_error.keys():
            if key != "time":  # Time should be summed, not averaged
                cumulative_error[key] /= num_materials

        # Store the cumulative error data for this calculator type
        composite_error_data[calculator_type] = cumulative_error

    # Once all materials and calculators have been processed, create a DataFrame
    composite_df = pd.DataFrame(composite_error_data).transpose()

    # Plot the composite scorecard
    plot_composite_scorecard(composite_df)

    # Save the composite dataframe
    composite_df.to_csv("composite_error_data.csv", index=True)

def plot_composite_scorecard(df):
    """Plot the composite scorecard for all calculators"""
    fig = px.imshow(df, text_auto=True, aspect="auto", labels=dict(color="Error"))
    fig.update_layout(title="Composite Scorecard for Calculators")
    
    # Save plot
    fname_plot = "composite_error_scorecard.png"
    fig.write_image(fname_plot)
    fig.show()

# Example usage
jid_list = ["JVASP-1002","JVASP-30"]
calculator_types = ["chgnet","mace"]

chemical_potentials_json = """{
    "Eu": {
        "jid": "JVASP-88846",
        "energy_alignn_ff": -1.959056,
        "energy_chgnet": -10.253422,
        "energy_matgl": -10.171406,
        "energy_mace": -10.2523955,
        "energy_sevennet": -10.2651205
    },
    "Ru": {
        "jid": "JVASP-987",
        "energy_alignn_ff": -4.911544,
        "energy_chgnet": -9.140766,
        "energy_matgl": -9.2708045,
        "energy_mace": -9.2447245,
        "energy_sevennet": -9.26455
    },
    "Re": {
        "jid": "JVASP-981",
        "energy_alignn_ff": -7.9723265,
        "energy_chgnet": -12.457021,
        "energy_matgl": -12.454343,
        "energy_mace": -12.443777,
        "energy_sevennet": -12.441499
    },
    "Rb": {
        "jid": "JVASP-25388",
        "energy_alignn_ff": 1.2639575,
        "energy_chgnet": -0.96709,
        "energy_matgl": -0.945327,
        "energy_mace": -0.947626,
        "energy_sevennet": -0.9447955
    },
    "Rh": {
        "jid": "JVASP-984",
        "energy_alignn_ff": -2.738144,
        "energy_chgnet": -7.300445,
        "energy_matgl": -7.390478,
        "energy_mace": -7.290643,
        "energy_sevennet": -7.341245
    },
    "Be": {
        "jid": "JVASP-834",
        "energy_alignn_ff": -1.172956,
        "energy_chgnet": -3.7293225,
        "energy_matgl": -3.6833285,
        "energy_mace": -3.7484385,
        "energy_sevennet": -3.7126385
    },
    "Ba": {
        "jid": "JVASP-14604",
        "energy_alignn_ff": 0.504432,
        "energy_chgnet": -1.882994,
        "energy_matgl": -1.884151,
        "energy_mace": -1.862737,
        "energy_sevennet": -1.823505
    },
    "Bi": {
        "jid": "JVASP-837",
        "energy_alignn_ff": -1.1796125,
        "energy_chgnet": -3.8214735,
        "energy_matgl": -3.8210865,
        "energy_mace": -3.8594375,
        "energy_sevennet": -3.860356
    },
    "Br": {
        "jid": "JVASP-840",
        "energy_alignn_ff": 0.111092,
        "energy_chgnet": -1.6219065,
        "energy_matgl": -1.57035975,
        "energy_mace": -1.918754,
        "energy_sevennet": -1.63592475
    },
    "H": {
        "jid": "JVASP-25379",
        "energy_alignn_ff": -3.390353,
        "energy_chgnet": -3.372895,
        "energy_matgl": -3.41707,
        "energy_mace": -3.35942325,
        "energy_sevennet": -3.3967095
    },
    "P": {
        "jid": "JVASP-25144",
        "energy_alignn_ff": -3.936694,
        "energy_chgnet": -5.403327,
        "energy_matgl": -5.35841075,
        "energy_mace": -5.41365725,
        "energy_sevennet": -5.3782965
    },
    "Os": {
        "jid": "JVASP-14744",
        "energy_alignn_ff": -6.7488375,
        "energy_chgnet": -11.17282,
        "energy_matgl": -11.2222565,
        "energy_mace": -11.208637,
        "energy_sevennet": -11.2106325
    },
    "Ge": {
        "jid": "JVASP-890",
        "energy_alignn_ff": -0.9296625,
        "energy_chgnet": -4.456685,
        "energy_matgl": -4.5874235,
        "energy_mace": -4.5258085,
        "energy_sevennet": -4.577574
    },
    "Ga": {
        "jid": "JVASP-14622",
        "energy_alignn_ff": 0.651836,
        "energy_chgnet": -3.01915925,
        "energy_matgl": -3.02938625,
        "energy_mace": -3.029211,
        "energy_sevennet": -3.0231815
    },
    "Pr": {
        "jid": "JVASP-969",
        "energy_alignn_ff": -2.2528295,
        "energy_chgnet": -4.7477165,
        "energy_matgl": -4.74495175,
        "energy_mace": -4.779047,
        "energy_sevennet": -4.763914
    },
    "Pt": {
        "jid": "JVASP-972",
        "energy_alignn_ff": -2.918793,
        "energy_chgnet": -6.092452,
        "energy_matgl": -6.084422,
        "energy_mace": -6.0348,
        "energy_sevennet": -6.059916
    },
    "Pu": {
        "jid": "JVASP-25254",
        "energy_alignn_ff": -9.9656085,
        "energy_chgnet": -14.2657575,
        "energy_matgl": -14.262664,
        "energy_mace": -14.1881525,
        "energy_sevennet": -14.259139
    },
    "C": {
        "jid": "JVASP-25407",
        "energy_alignn_ff": -6.86239725,
        "energy_chgnet": -9.180709,
        "energy_matgl": -9.15409575,
        "energy_mace": -9.21273225,
        "energy_sevennet": -9.21711925
    },
    "Pb": {
        "jid": "JVASP-961",
        "energy_alignn_ff": -0.271016,
        "energy_chgnet": -3.663821,
        "energy_matgl": -3.708089,
        "energy_mace": -3.696473,
        "energy_sevennet": -3.692711
    },
    "Pa": {
        "jid": "JVASP-958",
        "energy_alignn_ff": -6.305937,
        "energy_chgnet": -9.436422,
        "energy_matgl": -9.488003,
        "energy_mace": -9.385218,
        "energy_sevennet": -9.485789
    },
    "Pd": {
        "jid": "JVASP-963",
        "energy_alignn_ff": -1.464927,
        "energy_chgnet": -5.202854,
        "energy_matgl": -5.16607,
        "energy_mace": -5.187269,
        "energy_sevennet": -5.187177
    },
    "Cd": {
        "jid": "JVASP-14832",
        "energy_alignn_ff": 2.5374005,
        "energy_chgnet": -0.9251905,
        "energy_matgl": -0.937331,
        "energy_mace": -0.908682,
        "energy_sevennet": -0.906839
    },
    "Pm": {
        "jid": "JVASP-966",
        "energy_alignn_ff": -2.07879675,
        "energy_chgnet": -4.73003,
        "energy_matgl": -4.70232675,
        "energy_mace": -4.7350055,
        "energy_sevennet": -4.7350125
    },
    "Ho": {
        "jid": "JVASP-25125",
        "energy_alignn_ff": -1.8107043333333335,
        "energy_chgnet": -4.564541333333334,
        "energy_matgl": -4.5620140000000005,
        "energy_mace": -4.573492333333333,
        "energy_sevennet": -4.569997333333333
    },
    "Hf": {
        "jid": "JVASP-802",
        "energy_alignn_ff": -7.1547165,
        "energy_chgnet": -9.9135115,
        "energy_matgl": -9.94129,
        "energy_mace": -9.9602735,
        "energy_sevennet": -9.9526465
    },
    "Hg": {
        "jid": "JVASP-25273",
        "energy_alignn_ff": 2.640838,
        "energy_chgnet": -0.285384,
        "energy_matgl": -0.28593,
        "energy_mace": -0.288451,
        "energy_sevennet": -0.276385
    },
    "He": {
        "jid": "JVASP-25167",
        "energy_alignn_ff": 0.692646,
        "energy_chgnet": -0.051122,
        "energy_matgl": 0.004336,
        "energy_mace": 0.009653,
        "energy_sevennet": -0.0415045
    },
    "Mg": {
        "jid": "JVASP-919",
        "energy_alignn_ff": 1.408972,
        "energy_chgnet": -1.596146,
        "energy_matgl": -1.5950905,
        "energy_mace": -1.6067645,
        "energy_sevennet": -1.6045245
    },
    "K": {
        "jid": "JVASP-25114",
        "energy_alignn_ff": 1.250066,
        "energy_chgnet": -1.089017,
        "energy_matgl": -0.900627,
        "energy_mace": -1.088554,
        "energy_sevennet": -1.072364
    },
    "Mn": {
        "jid": "JVASP-922",
        "energy_alignn_ff": -4.3859348275862065,
        "energy_chgnet": -9.111478793103448,
        "energy_matgl": -9.133505999999999,
        "energy_mace": -9.164792827586208,
        "energy_sevennet": -9.115948896551725
    },
    "O": {
        "jid": "JVASP-949",
        "energy_alignn_ff": -3.092893625,
        "energy_chgnet": -4.93406625,
        "energy_matgl": -4.92246625,
        "energy_mace": -5.090092625,
        "energy_sevennet": -4.922183
    },
    "S": {
        "jid": "JVASP-95268",
        "energy_alignn_ff": -2.49916290625,
        "energy_chgnet": -4.15215875,
        "energy_matgl": -4.0800199375,
        "energy_mace": -4.11382196875,
        "energy_sevennet": -4.1272320625
    },
    "W": {
        "jid": "JVASP-79561",
        "energy_alignn_ff": -9.1716985,
        "energy_chgnet": -12.76264375,
        "energy_matgl": -12.88728225,
        "energy_mace": -12.74655525,
        "energy_sevennet": -12.7603645
    },
    "Zn": {
        "jid": "JVASP-1056",
        "energy_alignn_ff": 2.3596515,
        "energy_chgnet": -1.2631135,
        "energy_matgl": -1.279107,
        "energy_mace": -1.277153,
        "energy_sevennet": -1.2559245
    },
    "Zr": {
        "jid": "JVASP-14612",
        "energy_alignn_ff": -5.6843005,
        "energy_chgnet": -8.510046,
        "energy_matgl": -8.487857,
        "energy_mace": -8.5319785,
        "energy_sevennet": -8.5710905
    },
    "Er": {
        "jid": "JVASP-102277",
        "energy_alignn_ff": -1.8103855,
        "energy_chgnet": -4.54730325,
        "energy_matgl": -4.5462495,
        "energy_mace": -4.57055375,
        "energy_sevennet": -4.55634125
    },
    "Ni": {
        "jid": "JVASP-943",
        "energy_alignn_ff": 0.267769,
        "energy_chgnet": -5.746612,
        "energy_matgl": -5.740693,
        "energy_mace": -5.731884,
        "energy_sevennet": -5.768434
    },
    "Na": {
        "jid": "JVASP-931",
        "energy_alignn_ff": 0.90171,
        "energy_chgnet": -1.2953955,
        "energy_matgl": -1.290621,
        "energy_mace": -1.3133655,
        "energy_sevennet": -1.309338
    },
    "Nb": {
        "jid": "JVASP-934",
        "energy_alignn_ff": -6.71386,
        "energy_chgnet": -10.025986,
        "energy_matgl": -10.059839,
        "energy_mace": -10.085746,
        "energy_sevennet": -10.094292
    },
    "Nd": {
        "jid": "JVASP-937",
        "energy_alignn_ff": -2.18942225,
        "energy_chgnet": -4.7303625,
        "energy_matgl": -4.732285,
        "energy_mace": -4.77048025,
        "energy_sevennet": -4.751452
    },
    "Ne": {
        "jid": "JVASP-21193",
        "energy_alignn_ff": 2.326043,
        "energy_chgnet": -0.036456,
        "energy_matgl": -0.026325,
        "energy_mace": 0.154394,
        "energy_sevennet": -0.030487
    },
    "Fe": {
        "jid": "JVASP-25142",
        "energy_alignn_ff": -3.474067,
        "energy_chgnet": -8.3403385,
        "energy_matgl": -8.3962915,
        "energy_mace": -8.3867815,
        "energy_sevennet": -8.360154
    },
    "B": {
        "jid": "JVASP-828",
        "energy_alignn_ff": -5.147107083333333,
        "energy_chgnet": -6.5688657500000005,
        "energy_matgl": -6.606697083333334,
        "energy_mace": -6.587750083333333,
        "energy_sevennet": -6.631985333333334
    },
    "F": {
        "jid": "JVASP-33718",
        "energy_alignn_ff": 0.11726725,
        "energy_chgnet": -1.901216,
        "energy_matgl": -1.96192525,
        "energy_mace": -1.90583125,
        "energy_sevennet": -1.88403
    },
    "N": {
        "jid": "JVASP-25250",
        "energy_alignn_ff": -6.776678,
        "energy_chgnet": -8.35449975,
        "energy_matgl": -8.32737725,
        "energy_mace": -8.3102885,
        "energy_sevennet": -8.31863125
    },
    "Kr": {
        "jid": "JVASP-25213",
        "energy_alignn_ff": 1.967669,
        "energy_chgnet": -0.0637565,
        "energy_matgl": -0.0968225,
        "energy_mace": -0.060912,
        "energy_sevennet": -0.052547
    },
    "Si": {
        "jid": "JVASP-1002",
        "energy_alignn_ff": -4.0240705,
        "energy_chgnet": -5.3138255,
        "energy_matgl": -5.4190405,
        "energy_mace": -5.3420265,
        "energy_sevennet": -5.402981
    },
    "Sn": {
        "jid": "JVASP-14601",
        "energy_alignn_ff": -0.5207135,
        "energy_chgnet": -3.823353,
        "energy_matgl": -3.9790995,
        "energy_mace": -3.921247,
        "energy_sevennet": -3.9694585
    },
    "Sm": {
        "jid": "JVASP-14812",
        "energy_alignn_ff": -2.021082,
        "energy_chgnet": -4.688136,
        "energy_matgl": -4.671094,
        "energy_mace": -4.69254675,
        "energy_sevennet": -4.6989995
    },
    "V": {
        "jid": "JVASP-14837",
        "energy_alignn_ff": -4.914755,
        "energy_chgnet": -9.077443,
        "energy_matgl": -9.034538,
        "energy_mace": -9.098935,
        "energy_sevennet": -9.081524
    },
    "Sc": {
        "jid": "JVASP-996",
        "energy_alignn_ff": -3.340738,
        "energy_chgnet": -6.2751955,
        "energy_matgl": -6.2797465,
        "energy_mace": -6.297023,
        "energy_sevennet": -6.28785
    },
    "Sb": {
        "jid": "JVASP-993",
        "energy_alignn_ff": -2.089857,
        "energy_chgnet": -4.0672785,
        "energy_matgl": -4.1050825,
        "energy_mace": -4.0842765,
        "energy_sevennet": -4.1083495
    },
    "Se": {
        "jid": "JVASP-7804",
        "energy_alignn_ff": -1.7824116666666667,
        "energy_chgnet": -3.5392023333333333,
        "energy_matgl": -3.4793353333333332,
        "energy_mace": -3.4744116666666667,
        "energy_sevennet": -3.4769020000000004
    },
    "Co": {
        "jid": "JVASP-858",
        "energy_alignn_ff": -3.208796,
        "energy_chgnet": -7.0292245,
        "energy_matgl": -7.1024915,
        "energy_mace": -7.0844295,
        "energy_sevennet": -7.073042
    },
    "Cl": {
        "jid": "JVASP-25104",
        "energy_alignn_ff": -0.13556325,
        "energy_chgnet": -1.8968495,
        "energy_matgl": -1.8235,
        "energy_mace": -1.83524975,
        "energy_sevennet": -1.834533
    },
    "Ca": {
        "jid": "JVASP-25180",
        "energy_alignn_ff": 0.585537,
        "energy_chgnet": -1.970015,
        "energy_matgl": -1.98922,
        "energy_mace": -2.009631,
        "energy_sevennet": -1.991963
    },
    "Ce": {
        "jid": "JVASP-852",
        "energy_alignn_ff": -2.72069,
        "energy_chgnet": -5.862665,
        "energy_matgl": -5.862472,
        "energy_mace": -5.878778,
        "energy_sevennet": -5.897567
    },
    "Xe": {
        "jid": "JVASP-25248",
        "energy_alignn_ff": 2.3972965,
        "energy_chgnet": -0.026187,
        "energy_matgl": -0.1289395,
        "energy_mace": -0.023738,
        "energy_sevennet": -0.030133
    },
    "Tm": {
        "jid": "JVASP-1035",
        "energy_alignn_ff": -1.7555285,
        "energy_chgnet": -4.4662675,
        "energy_matgl": -4.447861,
        "energy_mace": -4.439375,
        "energy_sevennet": -4.4687315
    },
    "Cr": {
        "jid": "JVASP-861",
        "energy_alignn_ff": -5.394644,
        "energy_chgnet": -9.540979,
        "energy_matgl": -9.547915,
        "energy_mace": -9.450621,
        "energy_sevennet": -9.504951
    },
    "Cu": {
        "jid": "JVASP-867",
        "energy_alignn_ff": 1.481811,
        "energy_chgnet": -4.083517,
        "energy_matgl": -4.096651,
        "energy_mace": -4.08482,
        "energy_sevennet": -4.096196
    },
    "La": {
        "jid": "JVASP-910",
        "energy_alignn_ff": -2.45152875,
        "energy_chgnet": -4.8954715,
        "energy_matgl": -4.91215475,
        "energy_mace": -4.90372125,
        "energy_sevennet": -4.91955125
    },
    "Li": {
        "jid": "JVASP-25117",
        "energy_alignn_ff": -0.8245279999999999,
        "energy_chgnet": -1.8828183333333335,
        "energy_matgl": -1.9033153333333335,
        "energy_mace": -1.9064726666666667,
        "energy_sevennet": -1.9096840000000002
    },
    "Tl": {
        "jid": "JVASP-25337",
        "energy_alignn_ff": 0.869586,
        "energy_chgnet": -2.347048,
        "energy_matgl": -2.3508165,
        "energy_mace": -2.350755,
        "energy_sevennet": -2.345141
    },
    "Lu": {
        "jid": "JVASP-916",
        "energy_alignn_ff": -1.734765,
        "energy_chgnet": -4.4955295,
        "energy_matgl": -4.507209,
        "energy_mace": -4.515136,
        "energy_sevennet": -4.526551
    },
    "Ti": {
        "jid": "JVASP-1029",
        "energy_alignn_ff": -4.016277,
        "energy_chgnet": -7.7862876666666665,
        "energy_matgl": -7.875901666666667,
        "energy_mace": -7.816422666666667,
        "energy_sevennet": -7.815967666666666
    },
    "Te": {
        "jid": "JVASP-25210",
        "energy_alignn_ff": -1.177326,
        "energy_chgnet": -3.1693719999999996,
        "energy_matgl": -3.142266,
        "energy_mace": -3.099973,
        "energy_sevennet": -3.0926693333333333
    },
    "Tb": {
        "jid": "JVASP-1017",
        "energy_alignn_ff": -1.8727539999999998,
        "energy_chgnet": -4.597796,
        "energy_matgl": -4.619775333333333,
        "energy_mace": -4.620745,
        "energy_sevennet": -4.617845666666667
    },
    "Tc": {
        "jid": "JVASP-1020",
        "energy_alignn_ff": -6.328599,
        "energy_chgnet": -10.339349,
        "energy_matgl": -10.339732,
        "energy_mace": -10.3385925,
        "energy_sevennet": -10.3608055
    },
    "Ta": {
        "jid": "JVASP-1014",
        "energy_alignn_ff": -8.171007,
        "energy_chgnet": -11.851633,
        "energy_matgl": -11.877074,
        "energy_mace": -11.899179,
        "energy_sevennet": -11.837949
    },
    "Yb": {
        "jid": "JVASP-21197",
        "energy_alignn_ff": 1.062488,
        "energy_chgnet": -1.52423,
        "energy_matgl": -1.509347,
        "energy_mace": -1.511301,
        "energy_sevennet": -1.520443
    },
    "Dy": {
        "jid": "JVASP-870",
        "energy_alignn_ff": -1.8504466666666666,
        "energy_chgnet": -4.585288666666666,
        "energy_matgl": -4.583516666666667,
        "energy_mace": -4.602253666666667,
        "energy_sevennet": -4.587306666666667
    },
    "I": {
        "jid": "JVASP-895",
        "energy_alignn_ff": 0.43406375,
        "energy_chgnet": -1.5900875,
        "energy_matgl": -1.51894975,
        "energy_mace": -1.7035675,
        "energy_sevennet": -1.50836475
    },
    "U": {
        "jid": "JVASP-14725",
        "energy_alignn_ff": -7.5724695,
        "energy_chgnet": -11.169176,
        "energy_matgl": -11.181288,
        "energy_mace": -11.2126675,
        "energy_sevennet": -11.19569
    },
    "Y": {
        "jid": "JVASP-1050",
        "energy_alignn_ff": -3.85497,
        "energy_chgnet": -6.4445845,
        "energy_matgl": -6.436653,
        "energy_mace": -6.468465,
        "energy_sevennet": -6.450417
    },
    "Ac": {
        "jid": "JVASP-810",
        "energy_alignn_ff": -0.975209,
        "energy_chgnet": -4.06527375,
        "energy_matgl": -4.122096,
        "energy_mace": -4.11465225,
        "energy_sevennet": -4.0897685
    },
    "Ag": {
        "jid": "JVASP-14606",
        "energy_alignn_ff": 0.823297,
        "energy_chgnet": -2.81171,
        "energy_matgl": -2.805851,
        "energy_mace": -2.836523,
        "energy_sevennet": -2.827129
    },
    "Ir": {
        "jid": "JVASP-901",
        "energy_alignn_ff": -4.92212,
        "energy_chgnet": -8.842527,
        "energy_matgl": -8.896281,
        "energy_mace": -8.821177,
        "energy_sevennet": -8.861135
    },
    "Al": {
        "jid": "JVASP-816",
        "energy_alignn_ff": -1.937461,
        "energy_chgnet": -3.664113,
        "energy_matgl": -3.701539,
        "energy_mace": -3.728351,
        "energy_sevennet": -3.719476
    },
    "As": {
        "jid": "JVASP-14603",
        "energy_alignn_ff": -2.945886,
        "energy_chgnet": -4.6230125,
        "energy_matgl": -4.636945,
        "energy_mace": -4.6254615,
        "energy_sevennet": -4.6453695
    },
    "Ar": {
        "jid": "JVASP-819",
        "energy_alignn_ff": 1.947545,
        "energy_chgnet": -0.067081,
        "energy_matgl": -0.069127,
        "energy_mace": -0.061463,
        "energy_sevennet": -0.067557
    },
    "Au": {
        "jid": "JVASP-825",
        "energy_alignn_ff": -0.37486,
        "energy_chgnet": -3.235706,
        "energy_matgl": -3.261796,
        "energy_mace": -3.2462,
        "energy_sevennet": -3.266554
    },
    "In": {
        "jid": "JVASP-898",
        "energy_alignn_ff": 0.686736,
        "energy_chgnet": -2.699644,
        "energy_matgl": -2.71128,
        "energy_mace": -2.705124,
        "energy_sevennet": -2.715821
    },
    "Mo": {
        "jid": "JVASP-21195",
        "energy_alignn_ff": -7.134011,
        "energy_chgnet": -10.696192,
        "energy_matgl": -10.80138,
        "energy_mace": -10.694583,
        "energy_sevennet": -10.684041
    }
}"""

analyze_multiple_structures(jid_list, calculator_types, chemical_potentials_json)
