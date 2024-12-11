#!/usr/bin/env python
import os
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
from sklearn.metrics import mean_absolute_error
from chipsff.calcs import setup_calculator


class AlignnFFForcesAnalyzer:
    def __init__(
        self,
        calculator_type,
        output_dir=None,
        calculator_settings=None,
        num_samples=None,
    ):
        self.calculator_type = calculator_type
        self.output_dir = output_dir or f"alignn_ff_analysis_{calculator_type}"
        os.makedirs(self.output_dir, exist_ok=True)
        self.log_file = os.path.join(
            self.output_dir, "alignn_ff_analysis_log.txt"
        )
        self.setup_logger()
        self.calculator = setup_calculator(
            self.calculator_type, calculator_settings or {}
        )
        self.job_info = {
            "calculator_type": calculator_type,
        }
        self.num_samples = num_samples

    def setup_logger(self):
        self.logger = logging.getLogger("AlignnFFForcesAnalyzer")
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
        self.compare_alignn_ff_properties()

    def compare_alignn_ff_properties(self):
        """
        Compare forces and stresses calculated by the FF calculator with alignn_ff DFT data.
        """
        self.log("Loading alignn_ff_db dataset...")
        # Load the alignn_ff_db dataset
        alignn_ff_data = data("alignn_ff_db")
        self.log(f"Total entries in alignn_ff_db: {len(alignn_ff_data)}")

        # Initialize lists to store results
        force_results = []
        stress_results = []

        # Limit the number of samples if specified
        if self.num_samples:
            alignn_ff_data = alignn_ff_data[: self.num_samples]

        # Iterate over each entry
        for idx, entry in enumerate(alignn_ff_data):
            jid = entry.get("jid", f"structure_{idx}")
            atoms_dict = entry["atoms"]
            atoms = Atoms.from_dict(atoms_dict)
            dft_forces = np.array(entry["forces"])  # Assuming units of eV/Å
            dft_stresses = np.array(
                entry["stresses"]
            )  # Assuming units of eV/Å³

            # The 'stresses' in alignn_ff_db are in 3x3 format and units of eV/Å³
            # Convert DFT stresses from eV/Å³ to GPa for comparison
            dft_stresses_GPa = dft_stresses * -0.1  # kbar to GPa

            # Flatten the 3x3 stress tensor to a 9-component array for comparison
            dft_stress_flat = dft_stresses_GPa.flatten()

            # Calculate predicted properties
            predicted_forces, predicted_stresses = self.calculate_properties(
                atoms
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
                self.log(f"Skipping {jid}: Predicted stresses not available.")
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
                    "prediction": ";".join(map(str, predicted_stress_flat)),
                }
            )

            # Optional: Progress indicator
            if idx % 1000 == 0:
                self.log(
                    f"Processed {idx + 1}/{len(alignn_ff_data)} structures."
                )

        # Ensure we have data to process
        if not force_results or not stress_results:
            self.log("No valid data found. Exiting.")
            return

        # Save results to CSV files
        force_df = pd.DataFrame(force_results)
        force_csv = os.path.join(
            self.output_dir, f"AI-MLFF-forces-alignn_ff-test-multimae.csv"
        )
        force_df.to_csv(force_csv, index=False)
        self.log(f"Saved force comparison data to '{force_csv}'")

        stress_df = pd.DataFrame(stress_results)
        stress_csv = os.path.join(
            self.output_dir, f"AI-MLFF-stresses-alignn_ff-test-multimae.csv"
        )
        stress_df.to_csv(stress_csv, index=False)
        self.log(f"Saved stress comparison data to '{stress_csv}'")

        # Zip the CSV files
        self.zip_file(force_csv)
        self.zip_file(stress_csv)

        # Calculate error metrics
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
            target_forces, pred_forces, "Forces", "eV/Å", forces_plot_filename
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

    def calculate_properties(self, atoms):
        """
        Calculate forces and stresses on the given atoms.

        Returns:
            Tuple of forces and stresses.
        """
        # Convert atoms to ASE format and assign the calculator
        ase_atoms = atoms.ase_converter()
        ase_atoms.calc = self.calculator

        # Calculate properties
        forces = ase_atoms.get_forces()
        stresses = ase_atoms.get_stress()  # Voigt 6-component stress in eV/Å³

        return forces, stresses  # Return forces and stresses

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

    def zip_file(self, filename):
        zip_filename = filename + ".zip"
        with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(filename, arcname=os.path.basename(filename))
        os.remove(filename)  # Remove the original file
        self.log(f"Zipped data to '{zip_filename}'")

    def save_job_info(self):
        job_info_filename = os.path.join(
            self.output_dir, f"alignn_ff_{self.calculator_type}_job_info.json"
        )
        with open(job_info_filename, "w") as f:
            json.dump(self.job_info, f, indent=4)
