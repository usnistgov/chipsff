#!/usr/bin/env python
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from ase.units import kJ
from ase.stress import voigt_6_to_full_3x3_stress
from jarvis.db.figshare import data
from jarvis.core.atoms import Atoms
import pandas as pd
import logging
import zipfile
import plotly.express as px
from sklearn.metrics import mean_absolute_error
import argparse
from jarvis.db.jsonutils import loadjson
from chipsff.config import CHIPSFFConfig
from tqdm import tqdm
from chipsff.calcs import setup_calculator


# Ensure that the necessary modules and functions are imported
# from your existing codebase, such as `data`, `Atoms`, `voigt_6_to_full_3x3_stress`, etc.
# Example:
# from your_module import data, Atoms, voigt_6_to_full_3x3_stress, loadjson


class MPTrjAnalyzer:
    def __init__(
        self,
        calculator_type,
        output_dir=None,
        calculator_settings=None,
        num_samples=None,
    ):
        self.calculator_type = calculator_type
        self.output_dir = output_dir or f"mptrj_analysis_{calculator_type}"
        os.makedirs(self.output_dir, exist_ok=True)
        self.log_file = os.path.join(self.output_dir, "mptrj_analysis_log.txt")
        self.setup_logger()
        self.calculator = setup_calculator(
            self.calculator_type, calculator_settings or {}
        )
        self.job_info = {
            "calculator_type": calculator_type,
        }
        self.num_samples = num_samples

    def setup_logger(self):
        self.logger = logging.getLogger("MPTrjAnalyzer")
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(self.log_file)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.log(f"Logging initialized. Output directory: {self.output_dir}")

    def log(self, message):
        self.logger.info(message)
        print(message)

    def setup_calculator(self):
        self.log(f"Setting up calculator: {self.calculator_type}")
        return setup_calculator(self.calculator_type)

    def run(self):
        self.compare_mptrj_properties()

    def compare_mptrj_properties(self):
        """
        Compare forces and stresses calculated by the FF calculator with MP trajectory data.
        """
        self.log("Loading MP trajectory dataset...")
        try:
            # Load the MP trajectory dataset
            mptrj_data = data("m3gnet_mpf")
            self.log(f"Total entries in mptrj: {len(mptrj_data)}")
        except Exception as e:
            self.log(f"Failed to load MP trajectory dataset: {e}")
            return

        # Initialize lists to store results
        force_results = []
        stress_results = []

        # Limit the number of samples if specified
        if self.num_samples:
            mptrj_data = mptrj_data[: self.num_samples]
            self.log(f"Limiting analysis to first {self.num_samples} samples.")

        # Iterate over each entry with try/except to handle errors gracefully
        for idx, entry in enumerate(mptrj_data):
            jid = entry.get("jid", f"structure_{idx}")
            try:
                atoms_dict = entry["atoms"]
                atoms = Atoms.from_dict(atoms_dict)
                dft_forces = np.array(entry["force"])
                dft_stresses = np.array(entry["stress"])

                # Convert DFT stresses from eV/Å³ to GPa for comparison
                # Note: Ensure that the conversion factor is correct based on your data
                dft_stresses_GPa = dft_stresses * -0.1  # Example conversion

                # Flatten the 3x3 stress tensor to a 9-component array for comparison
                dft_stress_flat = dft_stresses_GPa.flatten()

                # Calculate predicted properties
                predicted_forces, predicted_stresses = (
                    self.calculate_properties(atoms)
                )

                # Handle predicted stresses
                if predicted_stresses is not None:
                    # Predicted stresses are in Voigt 6-component format and units of eV/Å³
                    # Convert to full 3x3 tensor
                    predicted_stress_tensor_eVA3 = voigt_6_to_full_3x3_stress(
                        predicted_stresses
                    )
                    # Convert to GPa
                    predicted_stresses_GPa = (
                        predicted_stress_tensor_eVA3 * 160.21766208
                    )  # eV/Å³ to GPa
                    # Flatten the tensor
                    predicted_stress_flat = predicted_stresses_GPa.flatten()
                else:
                    self.log(
                        f"Skipping {jid}: Predicted stresses not available."
                    )
                    continue  # Skip structures where stresses are not available

                # Store the results
                force_results.append(
                    {
                        "id": jid,
                        "target": ";".join(map(str, dft_forces.flatten())),
                        "prediction": ";".join(
                            map(str, predicted_forces.flatten())
                        ),
                    }
                )
                stress_results.append(
                    {
                        "id": jid,
                        "target": ";".join(map(str, dft_stress_flat)),
                        "prediction": ";".join(
                            map(str, predicted_stress_flat)
                        ),
                    }
                )

                # Optional: Progress indicator
                if (idx + 1) % 1000 == 0:
                    self.log(
                        f"Processed {idx + 1}/{len(mptrj_data)} structures."
                    )

            except Exception as e:
                self.log(f"Error processing {jid} at index {idx}: {e}")
                continue  # Continue with the next entry

        # Ensure we have data to process
        if not force_results or not stress_results:
            self.log("No valid data found for forces or stresses. Exiting.")
            return

        # Save results to CSV files
        try:
            force_df = pd.DataFrame(force_results)
            force_csv = os.path.join(
                self.output_dir, f"AI-MLFF-forces-mptrj-test-multimae.csv"
            )
            force_df.to_csv(force_csv, index=False)
            self.log(f"Saved force comparison data to '{force_csv}'")
        except Exception as e:
            self.log(f"Failed to save force comparison data: {e}")

        try:
            stress_df = pd.DataFrame(stress_results)
            stress_csv = os.path.join(
                self.output_dir, f"AI-MLFF-stresses-mptrj-test-multimae.csv"
            )
            stress_df.to_csv(stress_csv, index=False)
            self.log(f"Saved stress comparison data to '{stress_csv}'")
        except Exception as e:
            self.log(f"Failed to save stress comparison data: {e}")

        # Zip the CSV files
        self.zip_file(force_csv)
        self.zip_file(stress_csv)

        # Calculate error metrics
        try:
            # Forces MAE
            target_forces = np.concatenate(
                force_df["target"]
                .apply(lambda x: np.fromstring(x, sep=";"))
                .values
            )
            pred_forces = np.concatenate(
                force_df["prediction"]
                .apply(lambda x: np.fromstring(x, sep=";"))
                .values
            )
            forces_mae = mean_absolute_error(target_forces, pred_forces)
            self.log(f"Forces MAE: {forces_mae:.6f} eV/Å")

            # Stresses MAE
            target_stresses = np.concatenate(
                stress_df["target"]
                .apply(lambda x: np.fromstring(x, sep=";"))
                .values
            )
            pred_stresses = np.concatenate(
                stress_df["prediction"]
                .apply(lambda x: np.fromstring(x, sep=";"))
                .values
            )
            stresses_mae = mean_absolute_error(target_stresses, pred_stresses)
            self.log(f"Stresses MAE: {stresses_mae:.6f} GPa")

            # Save MAE to job_info
            self.job_info["forces_mae"] = forces_mae
            self.job_info["stresses_mae"] = stresses_mae
            self.save_job_info()

            # Plot parity plots
            forces_plot_filename = os.path.join(
                self.output_dir, f"forces_parity_plot.png"
            )
            self.plot_parity(
                target_forces,
                pred_forces,
                "Forces",
                "eV/Å",
                forces_plot_filename,
            )

            stresses_plot_filename = os.path.join(
                self.output_dir, f"stresses_parity_plot.png"
            )
            self.plot_parity(
                target_stresses,
                pred_stresses,
                "Stresses",
                "GPa",
                stresses_plot_filename,
            )

        except Exception as e:
            self.log(f"Error calculating error metrics: {e}")

    def calculate_properties(self, atoms):
        """
        Calculate forces and stresses on the given atoms.

        Returns:
            Tuple of forces and stresses.
        """
        try:
            # Convert atoms to ASE format and assign the calculator
            ase_atoms = atoms.ase_converter()
            ase_atoms.calc = self.calculator

            # Calculate properties
            forces = ase_atoms.get_forces()
            stresses = (
                ase_atoms.get_stress()
            )  # Voigt 6-component stress in eV/Å³

            return forces, stresses  # Return forces and stresses
        except Exception as e:
            self.log(f"Error calculating properties: {e}")
            return None, None

    def plot_parity(self, target, prediction, property_name, units, filename):
        """
        Plot parity plot for a given property.

        Args:
            target (array-like): Target values.
            prediction (array-like): Predicted values.
            property_name (str): Name of the property (e.g., 'Forces').
            units (str): Units of the property (e.g., 'eV/Å' or 'GPa').
            filename (str): Filename to save the plot.
        """
        try:
            plt.figure(figsize=(8, 8), dpi=300)
            plt.scatter(target, prediction, alpha=0.5, edgecolors="k", s=20)
            min_val = min(np.min(target), np.min(prediction))
            max_val = max(np.max(target), np.max(prediction))
            plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=2)
            plt.xlabel(f"Target {property_name} ({units})", fontsize=14)
            plt.ylabel(f"Predicted {property_name} ({units})", fontsize=14)
            plt.title(f"Parity Plot for {property_name}", fontsize=16)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()
            self.log(f"Saved parity plot for {property_name} as '{filename}'")
        except Exception as e:
            self.log(f"Error plotting parity for {property_name}: {e}")

    def zip_file(self, filename):
        try:
            if os.path.exists(filename):
                zip_filename = filename + ".zip"
                with zipfile.ZipFile(
                    zip_filename, "w", zipfile.ZIP_DEFLATED
                ) as zf:
                    zf.write(filename, arcname=os.path.basename(filename))
                os.remove(filename)  # Remove the original file
                self.log(f"Zipped data to '{zip_filename}'")
            else:
                self.log(
                    f"File '{filename}' does not exist. Skipping zipping."
                )
        except Exception as e:
            self.log(f"Error zipping file '{filename}': {e}")

    def save_job_info(self):
        try:
            job_info_filename = os.path.join(
                self.output_dir, f"mptrj_{self.calculator_type}_job_info.json"
            )
            with open(job_info_filename, "w") as f:
                json.dump(self.job_info, f, indent=4)
            self.log(f"Job info saved to '{job_info_filename}'")
        except Exception as e:
            self.log(f"Error saving job info: {e}")


class ScalingAnalyzer:
    def __init__(self, config):
        self.config = config
        self.scaling_numbers = config.scaling_numbers or [1, 2, 3, 4, 5]
        self.scaling_element = config.scaling_element or "Cu"
        self.scaling_calculators = config.scaling_calculators or [
            config.calculator_type
        ]
        self.calculator_settings = config.calculator_settings or {}
        elements_str = self.scaling_element
        self.output_dir = f"scaling_analysis_{elements_str}"
        os.makedirs(self.output_dir, exist_ok=True)
        self.log_file = os.path.join(
            self.output_dir, "scaling_analysis_log.txt"
        )
        self.setup_logger()
        self.job_info = {}

    def setup_logger(self):
        import logging

        self.logger = logging.getLogger("ScalingAnalyzer")
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(self.log_file)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.log(f"Logging initialized. Output directory: {self.output_dir}")

    def log(self, message):
        self.logger.info(message)
        print(message)

    def run(self):
        self.log("Starting scaling test...")
        from ase import Atoms, Atom
        from ase.build.supercells import make_supercell

        a = 3.6  # Lattice constant
        atoms = Atoms(
            [Atom(self.scaling_element, (0, 0, 0))],
            cell=0.5
            * a
            * np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]),
            pbc=True,
        )
        times_dict = {calc_type: [] for calc_type in self.scaling_calculators}
        natoms = []
        for i in self.scaling_numbers:
            self.log(f"Scaling test: Supercell size {i}")
            sc = make_supercell(atoms, [[i, 0, 0], [0, i, 0], [0, 0, i]])
            natoms.append(len(sc))
            for calc_type in self.scaling_calculators:
                # Setup calculator
                calc_settings = self.calculator_settings.get(calc_type, {})
                calculator = setup_calculator(calc_type, calc_settings)
                sc.calc = calculator
                # Measure time
                t1 = time.time()
                en = sc.get_potential_energy() / len(sc)
                t2 = time.time()
                times_dict[calc_type].append(t2 - t1)
                self.log(
                    f"Calculator {calc_type}: Time taken {t2 - t1:.4f} s for {len(sc)} atoms"
                )
        # Plot results
        plt.figure()
        for calc_type in self.scaling_calculators:
            plt.plot(natoms, times_dict[calc_type], "-o", label=calc_type)
        plt.xlabel("Number of atoms")
        plt.ylabel("Time (s)")
        plt.grid(True)
        plt.legend()
        scaling_plot_filename = os.path.join(
            self.output_dir, "scaling_test.png"
        )
        plt.savefig(scaling_plot_filename)
        plt.close()
        self.log(f"Scaling test plot saved to {scaling_plot_filename}")
        # Save results to job_info
        self.job_info["scaling_test"] = {"natoms": natoms, "times": times_dict}
        self.save_job_info()

    def save_job_info(self):
        job_info_filename = os.path.join(
            self.output_dir, "scaling_analysis_job_info.json"
        )
        with open(job_info_filename, "w") as f:
            json.dump(self.job_info, f, indent=4)
        self.log(f"Job info saved to '{job_info_filename}'")
