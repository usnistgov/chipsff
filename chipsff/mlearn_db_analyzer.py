#!/usr/bin/env python
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from ase.stress import voigt_6_to_full_3x3_stress
from jarvis.core.atoms import Atoms
import pandas as pd
import requests
import zipfile
from sklearn.metrics import mean_absolute_error
from chipsff.calcs import setup_calculator


class MLearnForcesAnalyzer:
    def __init__(
        self,
        calculator_type,
        mlearn_elements,
        output_dir=None,
        calculator_settings=None,
    ):
        self.calculator_type = calculator_type
        self.mlearn_elements = mlearn_elements
        elements_str = "_".join(self.mlearn_elements)
        self.output_dir = (
            output_dir or f"mlearn_analysis_{elements_str}_{calculator_type}"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        self.log_file = os.path.join(
            self.output_dir, "mlearn_analysis_log.txt"
        )
        self.setup_logger()
        self.calculator = setup_calculator(
            self.calculator_type, calculator_settings or {}
        )
        self.job_info = {
            "calculator_type": calculator_type,
            "mlearn_elements": mlearn_elements,
        }

    def setup_logger(self):
        import logging

        self.logger = logging.getLogger("MLearnForcesAnalyzer")
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(self.log_file)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def log(self, message):
        self.logger.info(message)
        print(message)

    def setup_calculator(self):
        return setup_calculator(self.calculator_type)

    def run(self):
        for element in self.mlearn_elements:
            self.compare_mlearn_properties(element)

    def compare_mlearn_properties(self, element):
        """
        Compare forces and stresses calculated by the FF calculator with mlearn DFT data for a given element.

        Args:
            element (str): Element symbol to filter structures (e.g., 'Si').
        """
        # Download the mlearn dataset if not already present
        mlearn_zip_path = "mlearn.json.zip"
        if not os.path.isfile(mlearn_zip_path):
            self.log("Downloading mlearn dataset...")
            url = "https://figshare.com/ndownloader/files/40357663"
            response = requests.get(url)
            with open(mlearn_zip_path, "wb") as f:
                f.write(response.content)
            self.log("Download completed.")

        # Read the JSON data from the zip file
        with zipfile.ZipFile(mlearn_zip_path, "r") as z:
            with z.open("mlearn.json") as f:
                mlearn_data = json.load(f)

        # Convert mlearn data to DataFrame
        df = pd.DataFrame(mlearn_data)

        # Filter the dataset for the specified element
        df["elements"] = df["atoms"].apply(lambda x: x["elements"])
        df = df[df["elements"].apply(lambda x: element in x)]
        df = df.reset_index(drop=True)
        self.log(
            f"Filtered dataset to {len(df)} entries containing element '{element}'"
        )

        # Initialize lists to store results
        force_results = []
        stress_results = []

        # Iterate over each structure
        for idx, row in df.iterrows():
            jid = row.get("jid", f"structure_{idx}")
            atoms_dict = row["atoms"]
            atoms = Atoms.from_dict(atoms_dict)
            dft_forces = np.array(row["forces"])
            dft_stresses = np.array(
                row["stresses"]
            )  # Original stresses in kBar

            # Convert DFT stresses from kBar to GPa
            dft_stresses_GPa = dft_stresses * 0.1  # kBar to GPa

            # Convert DFT stresses to full 3x3 tensors
            if dft_stresses_GPa.ndim == 1 and dft_stresses_GPa.size == 6:
                dft_stress_tensor = voigt_6_to_full_3x3_stress(
                    dft_stresses_GPa
                )
            else:
                self.log(
                    f"Skipping {jid}: DFT stresses not in expected format."
                )
                continue  # Skip structures with unexpected stress format

            # Calculate predicted properties
            predicted_forces, predicted_stresses = self.calculate_properties(
                atoms
            )

            # Convert predicted stresses from eV/Å³ to GPa
            if predicted_stresses is not None and predicted_stresses.size == 6:
                predicted_stresses_GPa = (
                    predicted_stresses * 160.21766208
                )  # eV/Å³ to GPa
                predicted_stress_tensor = voigt_6_to_full_3x3_stress(
                    predicted_stresses_GPa
                )
            else:
                self.log(f"Skipping {jid}: Predicted stresses not available.")
                continue  # Skip structures where stresses are not available

            # Flatten the 3x3 stress tensors to 9-component arrays for comparison
            dft_stress_flat = dft_stress_tensor.flatten()
            predicted_stress_flat = predicted_stress_tensor.flatten()

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
            if idx % 10 == 0:
                self.log(f"Processed {idx + 1}/{len(df)} structures.")

        # Ensure we have data to process
        if not force_results or not stress_results:
            self.log("No valid data found for forces or stresses. Exiting.")
            return

        # Save results to CSV files
        force_df = pd.DataFrame(force_results)
        force_csv = os.path.join(
            self.output_dir,
            f"AI-MLFF-forces-mlearn_{element}-test-multimae.csv",
        )
        force_df.to_csv(force_csv, index=False)
        self.log(f"Saved force comparison data to '{force_csv}'")

        stress_df = pd.DataFrame(stress_results)
        stress_csv = os.path.join(
            self.output_dir,
            f"AI-MLFF-stresses-mlearn_{element}-test-multimae.csv",
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
            .apply(lambda x: np.array(x.split(";"), dtype=float))
            .values
        )
        pred_forces = np.concatenate(
            force_df["prediction"]
            .apply(lambda x: np.array(x.split(";"), dtype=float))
            .values
        )
        forces_mae = mean_absolute_error(target_forces, pred_forces)
        self.log(f"Forces MAE for element '{element}': {forces_mae:.6f} eV/Å")

        # Stresses MAE
        target_stresses = np.concatenate(
            stress_df["target"]
            .apply(lambda x: np.array(x.split(";"), dtype=float))
            .values
        )
        pred_stresses = np.concatenate(
            stress_df["prediction"]
            .apply(lambda x: np.array(x.split(";"), dtype=float))
            .values
        )
        stresses_mae = mean_absolute_error(target_stresses, pred_stresses)
        self.log(
            f"Stresses MAE for element '{element}': {stresses_mae:.6f} GPa"
        )

        # Save MAE to job_info
        self.job_info[f"forces_mae_{element}"] = forces_mae
        self.job_info[f"stresses_mae_{element}"] = stresses_mae
        self.save_job_info()

        # Plot parity plots
        forces_plot_filename = os.path.join(
            self.output_dir, f"forces_parity_plot_{element}.png"
        )
        self.plot_parity(
            target_forces,
            pred_forces,
            "Forces",
            "eV/Å",
            forces_plot_filename,
            element,
        )

        stresses_plot_filename = os.path.join(
            self.output_dir, f"stresses_parity_plot_{element}.png"
        )
        self.plot_parity(
            target_stresses,
            pred_stresses,
            "Stresses",
            "GPa",
            stresses_plot_filename,
            element,
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
        stresses = ase_atoms.get_stress()  # Voigt 6-component stress

        return forces, stresses  # Return forces and stresses in Voigt notation

    def plot_parity(
        self, target, prediction, property_name, units, filename, element
    ):
        """
        Plot parity plot for a given property.

        Args:
            target (array-like): Target values.
            prediction (array-like): Predicted values.
            property_name (str): Name of the property (e.g., 'Forces').
            units (str): Units of the property (e.g., 'eV/Å' or 'GPa').
            filename (str): Filename to save the plot.
            element (str): Element symbol.
        """
        plt.figure(figsize=(8, 8), dpi=300)
        plt.scatter(target, prediction, alpha=0.5, edgecolors="k", s=20)
        min_val = min(np.min(target), np.min(prediction))
        max_val = max(np.max(target), np.max(prediction))
        plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=2)
        plt.xlabel(f"Target {property_name} ({units})", fontsize=14)
        plt.ylabel(f"Predicted {property_name} ({units})", fontsize=14)
        plt.title(
            f"Parity Plot for {property_name} - Element {element}", fontsize=16
        )
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
            self.output_dir, f"mlearn_{self.calculator_type}_job_info.json"
        )
        with open(job_info_filename, "w") as f:
            json.dump(self.job_info, f, indent=4)
