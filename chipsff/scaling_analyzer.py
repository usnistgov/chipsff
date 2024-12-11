#!/usr/bin/env python
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from chipsff.calcs import setup_calculator


class ScalingAnalyzer:
    def __init__(self, config):
        self.config = config
        self.scaling_numbers = config.scaling_numbers or [1, 2, 3, 4, 5]
        self.scaling_element = config.scaling_element or "Cu"
        self.scaling_calculators = (
            config.scaling_calculators or config.calculator_types
        )

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
