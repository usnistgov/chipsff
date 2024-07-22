# https://arxiv.org/pdf/2407.09674

from ase.optimize.fire import FIRE
from ase.constraints import ExpCellFilter
from jarvis.core.atoms import ase_to_atoms
from matgl.ext.ase import M3GNetCalculator
from alignn.ff.ff import AlignnAtomwiseCalculator, wt10_path
from chgnet.model.dynamics import CHGNetCalculator
from mace.calculators import mace_mp


class Evaluator(object):
    def __init__(
        atoms_dataset=[],
        calculator_type="",
        calculator=None,
        tasks=[
            "geometry_relaxation",
            "bulk_modulus",
            "vacancy_formation",
            "surface_formation",
            "zpe",
        ],
        fmax=0.05,
        nsteps=200,
    ):
        self.atoms_dataset = atoms_dataset
        self.calculator_type = calculator_name
        self.tasks = tasks
        self.fmax = fmax
        self.nsteps = nsteps
        self.calculator = calculator
        if self.calculator is None:
            self.calculator = setup_calculator()

        def setup_calculator(self):
            if self.calculator_type == "M3GNet":
                pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
                return M3GNetCalculator(pot)
            elif self.calculator_type == "ALIGNN-FF":
                model_path = default_path()
                return AlignnAtomwiseCalculator(path=model_path)
            elif calculator_type == "CHGNet":
                return CHGNetCalculator()
            elif calculator_type == "MACE":
                return mace_mp()
            else:
                raise ValueError("Unsupported calculator type")

        def general_relaxer(self,atoms=None):
            ase_atoms = atoms.ase_converter()
            ase_atoms.calc = self.calculator
            ase_atoms = ExpCellFilter(ase_atoms)
            dyn = FIRE(ase_atoms)
            dyn.run(fmax=self.fmax, steps=self.nsteps)
            en = ase_atoms.atoms.get_potential_energy()
            return en, ase_to_atoms(ase_atoms.atoms)
