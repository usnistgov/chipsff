# https://arxiv.org/pdf/2407.09674
import numpy as np
from ase.optimize.fire import FIRE
from ase.constraints import ExpCellFilter
from jarvis.core.atoms import ase_to_atoms
from jarvis.db.figshare import data, get_jid_data
from ase.calculators.emt import EMT
from jarvis.core.atoms import Atoms
from ase.eos import EquationOfState
from ase.units import kJ
from jarvis.analysis.structure.spacegroup import (
    Spacegroup3D,
    symmetrically_distinct_miller_indices,
)
from jarvis.analysis.defects.surface import Surface
from jarvis.analysis.defects.vacancy import Vacancy
from jarvis.analysis.thermodynamics.energetics import get_optb88vdw_energy
import ase
from jarvis.core.kpoints import Kpoints3D as Kpoints
from phonopy import Phonopy
from phonopy.file_IO import (
    write_FORCE_CONSTANTS,
)
import os
import matplotlib.pyplot as plt
from ase import Atoms as AseAtoms
from collections import defaultdict


class Evaluator(object):
    def __init__(
        self,
        atoms_dataset=None,
        calculator_type="",
        calculator=None,
        fmax=0.05,
        nsteps=200,
        max_miller_index=1,
        id_tag="jid",
        tasks=[
            "geometry_relaxation",
            "bulk_modulus",
            "vacancy_formation",
            "surface_formation",
            "zpe",
        ],
        chem_pot_ref={},
        alignn_model_path=None,
        output_dir="out",
    ):
        self.atoms_dataset = atoms_dataset
        if isinstance(atoms_dataset, str):
            self.atoms_dataset = data(atoms_dataset)
        self.calculator_type = calculator_type
        self.tasks = tasks
        self.fmax = fmax
        self.nsteps = nsteps
        self.calculator = calculator
        self.id_tag = id_tag
        self.alignn_model_path = alignn_model_path
        self.max_miller_index = max_miller_index
        if self.calculator is None:
            self.calculator = self.setup_calculator()
        if not chem_pot_ref:
            chem_pot_ref = defaultdict()
        self.chem_pot_ref = chem_pot_ref
        self.output_dir = output_dir

    # TODO: import from intermat
    # https://github.com/usnistgov/intermat/blob/main/intermat/calculators.py

    def setup_calculator(self):
        if self.calculator_type == "emt":
            return EMT()
        elif self.calculator_type == "M3GNet":
            from matgl.ext.ase import M3GNetCalculator

            pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        elif self.calculator_type == "ALIGNN-FF":
            from alignn.ff.ff import AlignnAtomwiseCalculator, default_path

            # modl_path = "/wrk/knc6/AFFBench/aff307k_lmdb_param_low_rad_use_force_mult_mp/out111continue5"
            # model_path = "aff307k_lmdb_param_low_rad_use_cutoff_take4_noforce_mult/out111"
            # model_path = "/wrk/knc6/AFFBench/aff307k_lmdb_param_low_rad_use_force_mult_mp_tak4/out111a"
            # model_path = "/wrk/knc6/AFFBench/fd2.5mil_lmdb_param_low_rad_use_cutoff_take4_noforce_mult/out111"
            if self.alignn_model_path is None:

                model_path = default_path()
            else:
                model_path = self.alignn_model_path
            return AlignnAtomwiseCalculator(
                # path=model_path, stress_wt=0.3, model_filename='best_model.pt',force_mult_natoms=False
                path=model_path,
                stress_wt=0.3,
                model_filename="current_model.pt",
                force_mult_natoms=False,
            )
        elif self.calculator_type == "CHGNet":
            from chgnet.model.dynamics import CHGNetCalculator

            return CHGNetCalculator()
        elif self.calculator_type == "MACE":
            from mace.calculators import mace_mp

            return mace_mp()
        else:
            raise ValueError("Unsupported calculator type")

    def get_energy(
        self, atoms=None, cell_relax=True, constant_volume=False, id=None
    ):
        ase_atoms = atoms.ase_converter()
        ase_atoms.calc = self.calculator
        if not cell_relax:
            en = ase_atoms.get_potential_energy()
            return en, atoms
        ase_atoms = ExpCellFilter(ase_atoms, constant_volume=constant_volume)
        # TODO: Make it work with any other optimizer
        dyn = FIRE(ase_atoms)
        dyn.run(fmax=self.fmax, steps=self.nsteps)
        en = ase_atoms.atoms.get_potential_energy()
        final_atoms = ase_to_atoms(ase_atoms.atoms)
        return en, final_atoms

    def run_all(self):
        for i in self.atoms_dataset:
            id = i[self.id_tag]
            atoms = Atoms.from_dict(i["atoms"])
            print("atoms", atoms)
            energy, optim_atoms = self.get_energy(atoms=atoms, id=id)
            print("final atoms", optim_atoms)
            elastic_tensor = self.elastic_tensor(atoms=optim_atoms, id=id)
            kv = self.ev_curve(atoms=optim_atoms, id=id)
            surface_energy = self.surface_energy(atoms=optim_atoms, id=id)
            vacancy_energy = self.vacancy_energy(atoms=optim_atoms, id=id)
            print("energy, kv,elastic_tensor", energy, kv, elastic_tensor)
            self.phonons(atoms=optim_atoms, id=id)

    def elastic_tensor(self, atoms=None, id=None):
        from elastic import get_elementary_deformations, get_elastic_tensor
        import elastic

        ase_atoms = atoms.ase_converter()
        ase_atoms.calc = self.calculator
        systems = get_elementary_deformations(ase_atoms)
        cij_order = elastic.elastic.get_cij_order(ase_atoms)
        Cij, Bij = get_elastic_tensor(ase_atoms, systems)
        c11 = ""
        for i, j in zip(cij_order, Cij):
            print(i, j / ase.units.GPa)
            if i == "C_11":
                c11 = j / ase.units.GPa
        return c11

    def surface_energy(self, atoms=None, jid="x", cell_relax=False, id=None):
        spg = Spacegroup3D(atoms=atoms)
        cvn = spg.conventional_standard_structure
        mills = symmetrically_distinct_miller_indices(
            max_index=self.max_miller_index, cvn_atoms=cvn
        )
        bulk_enp = (
            self.get_energy(
                atoms=cvn, cell_relax=cell_relax, constant_volume=False
            )[0]
            / cvn.num_atoms
        )
        surface_results = []
        for j in mills:
            surf = Surface(
                atoms=cvn, indices=j, thickness=25, vacuum=15
            ).make_surface()
            name = (
                str(jid)
                + "_"
                + str(surf.composition.reduced_formula)
                + "_"
                + str("_".join(map(str, j)))
            )
            m = surf.lattice.matrix
            area = np.linalg.norm(np.cross(m[0], m[1]))
            surf_en = (
                16
                * (
                    self.get_energy(
                        atoms=surf,
                        cell_relax=cell_relax,
                        constant_volume=False,
                    )[0]
                    - bulk_enp * surf.num_atoms
                )
                / (2 * area)
            )
            info = {}
            info["name"] = name
            info["surf_en"] = surf_en
            print(name, surf_en)
            surface_results.append(info)
        return surface_results

    def vacancy_energy(
        self,
        atoms=None,
        jid="x",
        cell_relax=False,
        id=None,
    ):
        chem_pot = get_optb88vdw_energy()
        strts = Vacancy(atoms).generate_defects(
            on_conventional_cell=True, enforce_c_size=10, extend=1
        )
        bulk_enp = (
            self.get_energy(
                atoms=atoms, cell_relax=cell_relax, constant_volume=False
            )[0]
            / atoms.num_atoms
        )
        vacancy_results = []
        for j in strts:
            strt = Atoms.from_dict(
                j.to_dict()["defect_structure"]
            ).center_around_origin()
            name = (
                str(jid)
                + "_"
                + str(strt.composition.reduced_formula)
                + "_"
                + j.to_dict()["symbol"]
                + "_"
                + j.to_dict()["wyckoff_multiplicity"]
            )

            if j.to_dict()["symbol"] in self.chem_pot_ref:
                atoms_en = self.chem_pot_ref[j.to_dict()["symbol"]]
            else:
                jid_elemental = chem_pot[j.to_dict()["symbol"]]["jid"]
                atoms_elemental = Atoms.from_dict(
                    get_jid_data(jid=jid_elemental, dataset="dft_3d")["atoms"]
                )
                atoms_en = (
                    self.get_energy(
                        atoms=atoms_elemental,
                        cell_relax=cell_relax,
                        constant_volume=False,
                    )[0]
                ) / atoms_elemental.num_atoms
                self.chem_pot_ref[j.to_dict()["symbol"]] = atoms_en
            defect_en = (
                self.get_energy(
                    atoms=strt, cell_relax=cell_relax, constant_volume=False
                )[0]
                - bulk_enp * strt.num_atoms
                + atoms_en
            )
            info = {}
            info["name"] = name
            info["defect_en"] = defect_en
            print(name, defect_en)
            vacancy_results.append(info)
        return vacancy_results

    def phonons(
        self,
        atoms=None,
        dim=[2, 2, 2],
        freq_conversion_factor=33.3566830,  # Thz to cm-1
        write_fc=False,
        min_freq_tol=-0.05,
        distance=0.2,
        id=None,
    ):
        """Make Phonon calculation setup."""
        if atoms is None or self.calculator is None:
            raise ValueError("Atoms and calculator must be provided")

        kpoints = Kpoints().kpath(atoms, line_density=5)
        bulk = atoms.phonopy_converter()
        phonon = Phonopy(
            bulk, [[dim[0], 0, 0], [0, dim[1], 0], [0, 0, dim[2]]]
        )
        phonon.generate_displacements(distance=distance)
        supercells = phonon.get_supercells_with_displacements()
        set_of_forces = []
        for scell in supercells:
            ase_atoms = AseAtoms(
                symbols=scell.symbols,
                positions=scell.positions,
                cell=scell.cell,
                pbc=True,
            )
            ase_atoms.calc = self.calculator
            forces = np.array(ase_atoms.get_forces())
            drift_force = forces.sum(axis=0)
            for force in forces:
                force -= drift_force / forces.shape[0]
            set_of_forces.append(forces)

        phonon.produce_force_constants(forces=set_of_forces)
        if write_fc:
            write_FORCE_CONSTANTS(
                phonon.get_force_constants(), filename="FORCE_CONSTANTS"
            )

        lbls = kpoints.labels
        lbls_ticks = []
        freqs = []
        tmp_kp = []
        lbls_x = []
        count = 0
        for ii, k in enumerate(kpoints.kpts):
            k_str = ",".join(map(str, k))
            if ii == 0:
                tmp = []
                for i, freq in enumerate(phonon.get_frequencies(k)):
                    tmp.append(freq)
                freqs.append(tmp)
                tmp_kp.append(k_str)
                lbl = "$" + str(lbls[ii]) + "$"
                lbls_ticks.append(lbl)
                lbls_x.append(count)
                count += 1
            elif k_str != tmp_kp[-1]:
                tmp_kp.append(k_str)
                tmp = []
                for i, freq in enumerate(phonon.get_frequencies(k)):
                    tmp.append(freq)
                freqs.append(tmp)
                lbl = lbls[ii]
                if lbl != "":
                    lbl = "$" + str(lbl) + "$"
                    lbls_ticks.append(lbl)
                    lbls_x.append(count)
                count += 1

        freqs = np.array(freqs)
        freqs = freqs * freq_conversion_factor
        the_grid = plt.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.0)
        plt.rcParams.update({"font.size": 18})
        plt.figure(figsize=(10, 5))
        plt.subplot(the_grid[0])
        for i in range(freqs.shape[1]):
            plt.plot(freqs[:, i], lw=2, c="b")
        for i in lbls_x:
            plt.axvline(x=i, c="black")
        plt.xticks(lbls_x, lbls_ticks)
        plt.ylabel("Frequency (cm)")
        plt.xlim([0, max(lbls_x)])

        phonon.run_mesh(
            [20, 20, 20], is_gamma_center=True, is_mesh_symmetry=False
        )
        phonon.run_thermal_properties(t_step=10, t_max=1000, t_min=0)
        tprop_dict = phonon.get_thermal_properties_dict()
        zpe = (
            tprop_dict["free_energy"][0] * 0.0103643
        )  # converting from kJ/mol to eV
        phonon.run_total_dos()
        tdos = phonon._total_dos
        # freqs, ds = tdos.get_dos()
        freqs = tdos.frequency_points
        ds = tdos.dos
        freqs = np.array(freqs)

        freqs = freqs * freq_conversion_factor
        min_freq = min_freq_tol * freq_conversion_factor
        max_freq = max(freqs)
        plt.ylim([min_freq, max_freq])

        plt.subplot(the_grid[1])
        plt.fill_between(
            ds, freqs, color=(0.2, 0.4, 0.6, 0.6), edgecolor="k", lw=1, y2=0
        )
        plt.xlabel("DOS")
        plt.yticks([])
        plt.xticks([])
        plt.ylim([min_freq, max_freq])
        plt.xlim([0, max(ds)])
        plt.tight_layout()
        nm = id + "_phonon.png"
        phonopy_bands_figname = os.path.join(self.output_dir, nm)
        plt.savefig(phonopy_bands_figname)
        plt.show()
        plt.close()

        return zpe, phonon

    def ev_curve(self, atoms=None, dx=np.arange(-0.05, 0.05, 0.005), id=None):
        """Get EV curve."""

        y = []
        vol = []
        for i in dx:
            s1 = atoms.strain_atoms(i)
            ase_atoms = s1.ase_converter()
            ase_atoms.calc = self.calculator
            energy = ase_atoms.get_potential_energy()
            y.append(energy)
            vol.append(s1.volume)
        x = np.array(dx)
        y = np.array(y)
        kv = 'na'
        try:
         eos = EquationOfState(vol, y, eos="murnaghan")
         v0, e0, B = eos.fit()
         kv = B / kJ * 1.0e24  # , 'GPa')
        except:
           pass
        # print("Energies:", y)
        # print("Volumes:", vol)
        nm = id + "_eV.png"
        plt.plot(vol, y, "-o")
        fname = self.output_dir + "/" + nm
        plt.savefig(fname)
        plt.close()

        return kv


if __name__ == "__main__":
    jids_check = [
        "JVASP-1002",  # Si
        "JVASP-816",  # Al
        "JVASP-867",  # Cu
        "JVASP-1029",  # Ti
        "JVASP-861",  # Cr
        "JVASP-30",  # GaN Pg3mmc
        "JVASP-8169",  # GaN F-43m
        "JVASP-890",  # Ge
        "JVASP-8158",  # SiC F-43m
        "JVASP-8118",  # SiC P6_3mmc
        "JVASP-107",  # SiC P6_3mc
        "JVASP-39",  # AlN P6_3mc
        "JVASP-7844",  # AlN F-43m
        "JVASP-35106",  # Al3GaN4 P-43m
        "JVASP-1174",  # GaAs F-43m
        "JVASP-1372",  # AlAs F-43m
        "JVASP-91",  # C Fd-3m
        "JVASP-1186",  # InAs F-43M
        "JVASP-1408",  # AlSb F-43M
        "JVASP-105410",  # SiGe F-43m
        "JVASP-1177",  # GaSb F-43m
        "JVASP-79204",  # BN P63mc
        "JVASP-1393",  # GaP F-43m
        "JVASP-1312",  # BP F-43m
        "JVASP-1327",  # AlP F-43m
        "JVASP-1183",  # InP F-43m
        "JVASP-1192",  # CdSe F-43m
        "JVASP-8003",  # CdS F-43m
        "JVASP-96",  # ZnSe F-43m
        "JVASP-1198",  # ZnTe F-43m
        "JVASP-1195",  # ZnO P63mc
        "JVASP-9147",  # HfO2 P21c
        "JVASP-41",  # SiO2 P3_221
        "JVASP-34674",  # SiO2 C222_1
        "JVASP-113",  # ZrO2 P2_1c
        "JVASP-32",  # Al2O3 R-3c
    ]
    for i in jids_check:
        print(i)
        # atoms = Atoms.from_poscar("POSCAR").to_dict()
        atoms = (get_jid_data(jid=i, dataset="dft_3d")['atoms'])
        ev = Evaluator(
            atoms_dataset=[{"atoms": atoms, "jid": i}],
            # calculator_type="CHGNet",
            calculator_type="ALIGNN-FF",
            # calculator_type="emt",
            # alignn_model_path = None
            #alignn_model_path="/wrk/knc6/AFFBench/aff307k_lmdb_param_low_rad_use_force_mult_mp_tak4/out111b",
            #alignn_model_path = "/wrk/knc6/AFFBench/aff307k_lmdb_param_low_rad_use_cutoff_take4_noforce_mult/out111e/"
            alignn_model_path = "/wrk/knc6/AFFBench/fd2.5mil_lmdb_param_low_rad_use_cutoff_take4_noforce_mult/out111a"
        )
        ev.run_all()
