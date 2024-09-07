import numpy as np
from ase.optimize.fire import FIRE
from ase.constraints import ExpCellFilter
from jarvis.core.atoms import ase_to_atoms
from jarvis.db.figshare import data, get_jid_data, get_request_data
from ase.calculators.emt import EMT
from jarvis.core.atoms import Atoms
from ase.eos import EquationOfState
from ase.units import kJ
from jarvis.analysis.structure.spacegroup import (
    Spacegroup3D,
    symmetrically_distinct_miller_indices,
)
from jarvis.db.jsonutils import dumpjson
from jarvis.analysis.defects.surface import Surface
from jarvis.analysis.defects.vacancy import Vacancy
from jarvis.analysis.thermodynamics.energetics import get_optb88vdw_energy
from intermat.generate import InterfaceCombi
import ase
from jarvis.core.kpoints import Kpoints3D as Kpoints
from phonopy import Phonopy
from phonopy.file_IO import (
    write_FORCE_CONSTANTS,
)
import pandas as pd
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from ase import Atoms as AseAtoms
from collections import defaultdict
import glob
from jarvis.db.jsonutils import loadjson
from sklearn.metrics import mean_absolute_error
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import r2_score
from alignn.ff.ff import AlignnAtomwiseCalculator, default_path


dft_3d = data("dft_3d")
y = data("vacancydb")
surf_url = "https://figshare.com/ndownloader/files/46355689"
z = get_request_data(js_tag="surface_db_dd.json", url=surf_url)
defect_ids = list(set([i["jid"] for i in y]))
surf_ids = list(
    set([i["name"].split("Surface-")[1].split("_miller_")[0] for i in z])
)
mem = []
for i in dft_3d:
    tmp = i
    tmp["vacancy"] = {}
    tmp["surface"] = {}
    if i["jid"] in defect_ids:
        for j in y:
            if i["jid"] == j["jid"]:
                tmp["vacancy"].setdefault(
                    j["id"].split("_")[0] + "_" + j["id"].split("_")[1],
                    j["ef"],
                )
                # tmp["vacancy"].setdefault(j["id"], j["ef"])
                # Ignoring wyckoff notation
    if i["jid"] in surf_ids:
        for k in z:
            jid = k["name"].split("Surface-")[1].split("_miller_")[0]
            if i["jid"] == jid:
                # tmp['surface'].setdefault("_".join(k["name"].split('_')[0:5]), k["surf_en"])
                tmp["surface"].setdefault(
                    "_".join(k["name"].split("_thickness")[0].split("_")[0:5]),
                    k["surf_en"],
                )

    mem.append(tmp)

dft_3d = mem


def get_entry(jid):
    for i in dft_3d:
        if i["jid"] == jid:
            return i


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
        make_plot=True,
        chem_pot_relax=False,
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
        self.make_plot = make_plot
        self.chem_pot_relax = chem_pot_relax
        if self.calculator is None:
            self.calculator = self.setup_calculator()
        if not chem_pot_ref:
            chem_pot_ref = defaultdict()
        self.chem_pot_ref = chem_pot_ref
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # TODO: import from intermat
    # https://github.com/usnistgov/intermat/blob/main/intermat/calculators.py

    def setup_calculator(self):
        if self.calculator_type == "emt":
            return EMT()
        elif self.calculator_type == "M3GNet":
            import matgl
            from matgl.ext.ase import M3GNetCalculator

            pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
            calculator = M3GNetCalculator(pot, stress_weight=0.01)
            return calculator
        elif self.calculator_type == "ALIGNN-FF":
            from alignn.ff.ff import AlignnAtomwiseCalculator, default_path

            # modl_path = "/wrk/knc6/AFFBench/aff307k_lmdb_param_low_rad_use_force_mult_mp/out111continue5"
            # model_path = "aff307k_lmdb_param_low_rad_use_cutoff_take4_noforce_mult/out111"
            if self.alignn_model_path is None:

                model_path = default_path()
            else:
                model_path = self.alignn_model_path
            return AlignnAtomwiseCalculator(
                path=model_path,
                stress_wt=0.3,
                model_filename="current_model.pt",
                # model_filename="best_model.pt",
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
        self,
        atoms=None,
        cell_relax=True,
        constant_volume=False,
        id=None,
        entry=None,
    ):
        t1 = time.time()
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
        t2 = time.time()
        info = {}
        initial_abc = atoms.lattice.abc
        info["initial_a"] = initial_abc[0]
        info["initial_b"] = initial_abc[1]
        info["initial_c"] = initial_abc[2]
        info["initial_vol"] = atoms.volume
        final_abc = final_atoms.lattice.abc

        info["final_a"] = final_abc[0]
        info["final_b"] = final_abc[1]
        info["final_c"] = final_abc[2]
        info["final_vol"] = final_atoms.volume
        print("latt a", initial_abc[0], final_abc[0])
        print("latt b", initial_abc[1], final_abc[1])
        print("latt c", initial_abc[2], final_abc[2])
        info["energy"] = en
        # if id is not None:
        #    name = id + "_energy.json"
        #    fname = os.path.join(self.output_dir, name)
        #    dumpjson(data=info, filename=fname)
        return en, final_atoms, info

    def get_formation_energy(
        self,
        atoms=None,
        id=None,
        energy=None,
        cell_relax=True,
        entry=None,
        entry_key="formation_energy_peratom",
    ):

        if energy is None:
            energy, optim_atoms = self.get_energy(atoms=atoms, id=id)

        chem_pot = get_optb88vdw_energy()

        ase_atoms = atoms.ase_converter()
        form_energy = energy
        for i, j in atoms.composition.to_dict().items():

            if i in self.chem_pot_ref:
                atoms_en = self.chem_pot_ref[i]
            else:
                jid_elemental = chem_pot[i]["jid"]
                atoms_elemental = Atoms.from_dict(
                    get_entry(jid_elemental)["atoms"]
                )
                #    get_jid_data(jid=jid_elemental, dataset="dft_3d")["atoms"]
                # )
                atoms_en = (
                    self.get_energy(
                        atoms=atoms_elemental,
                        cell_relax=self.chem_pot_relax,
                        constant_volume=False,
                    )[0]
                ) / atoms_elemental.num_atoms
                self.chem_pot_ref[i] = atoms_en
                # print('i,j,en',i,j,atoms_en)
                form_energy = form_energy - atoms_en * j
        form_energy = form_energy / atoms.num_atoms
        form_energy_entry = ""
        if entry is not None and entry_key in entry:
            form_energy_entry = entry[entry_key]
        info = {}
        info["form_energy"] = form_energy
        info["form_energy_entry"] = form_energy_entry
        print("form_energy", form_energy, form_energy_entry)
        # name = id + "_form_energy.json"
        # fname = os.path.join(self.output_dir, name)
        # dumpjson(data=info, filename=fname)
        return info

    def run_all(self):
        do_phonon = True
        initial_a = []
        initial_b = []
        initial_c = []
        initial_vol = []
        final_a = []
        final_b = []
        final_c = []
        final_vol = []
        form_en = []
        form_en_entry = []
        c11 = []
        c11_entry = []
        c44 = []
        c44_entry = []
        kv = []
        kv_entry = []
        surf_en = []
        surf_en_entry = []
        vac_en = []
        vac_en_entry = []
        all_dat = {}
        ids = []
        wad_entry = []
        wad = []
        timings = []
        for i in tqdm(self.atoms_dataset, total=len(self.atoms_dataset)):
            t1 = time.time()
            tmp = {}
            id = i[self.id_tag]
            ids.append(id)
            if "entry" in i:
                entry = i["entry"]
            atoms = Atoms.from_dict(i["atoms"])
            print("id", id)
            print("atoms", atoms)
            energy, optim_atoms, info = self.get_energy(
                atoms=atoms, id=id, entry=entry
            )
            tmp["energy"] = info
            print("final atoms", optim_atoms)
            fen = self.get_formation_energy(
                atoms=optim_atoms, energy=energy, id=id, entry=entry
            )
            tmp["form_en"] = fen
            # print('form_en',form_en)
            elastic_tensor = self.elastic_tensor(
                atoms=optim_atoms, id=id, entry=entry
            )
            tmp["elastic_tensor"] = elastic_tensor
            modulus = self.ev_curve(atoms=optim_atoms, id=id, entry=entry)
            tmp["modulus"] = modulus
            surface_energy = self.surface_energy(
                atoms=optim_atoms, id=id, entry=entry
            )
            for ii in surface_energy:
                if ii["surf_en_entry"] != "":
                    surf_en_entry.append(ii["surf_en_entry"])
                    surf_en.append(ii["surf_en"])
            tmp["surface_energy"] = surface_energy
            vacancy_energy = self.vacancy_energy(
                atoms=optim_atoms, id=id, entry=entry
            )
            for ii in vacancy_energy:
                if ii["vac_en_entry"] != "":
                    vac_en_entry.append(ii["vac_en_entry"])
                    vac_en.append(ii["vac_en"])
            tmp["vacancy_energy"] = vacancy_energy
            if do_phonon:
                print("Running phonon")
                phon = self.phonons(atoms=optim_atoms, id=id)
            # tmp['phon']=phon
            name = id + "_dat.json"
            fname = os.path.join(self.output_dir, name)
            dumpjson(data=tmp, filename=fname)
            d = tmp
            initial_a.append(d["energy"]["initial_a"])
            final_a.append(d["energy"]["final_a"])
            initial_b.append(d["energy"]["initial_b"])
            final_b.append(d["energy"]["final_b"])
            initial_c.append(d["energy"]["initial_c"])
            final_c.append(d["energy"]["final_c"])
            initial_vol.append(d["energy"]["initial_vol"])
            final_vol.append(d["energy"]["final_vol"])
            if d["elastic_tensor"]["c11_entry"] != "":
                c11.append(d["elastic_tensor"]["c11"])
                c11_entry.append(d["elastic_tensor"]["c11_entry"])
            if d["elastic_tensor"]["c44_entry"] != "":
                c44.append(d["elastic_tensor"]["c44"])
                c44_entry.append(d["elastic_tensor"]["c44_entry"])
            if d["modulus"]["kv_entry"] != "":
                kv.append(d["modulus"]["kv"])
                kv_entry.append(d["modulus"]["kv_entry"])
            if d["form_en"]["form_energy_entry"] != "":
                form_en.append(d["form_en"]["form_energy"])
                form_en_entry.append(d["form_en"]["form_energy_entry"])
            t2 = time.time()
            timings.append(t2 - t1)
            print("time", t2 - t1)
        all_dat = {}
        all_dat["ids"] = ids
        all_dat["initial_a"] = initial_a
        all_dat["initial_b"] = initial_b
        all_dat["initial_c"] = initial_c
        all_dat["final_a"] = final_a
        all_dat["final_b"] = final_b
        all_dat["final_c"] = final_c
        all_dat["initial_vol"] = initial_vol
        all_dat["final_vol"] = final_vol
        all_dat["c11"] = c11
        all_dat["c11_entry"] = c11_entry
        all_dat["c44"] = c44
        all_dat["c44_entry"] = c44_entry
        all_dat["kv"] = kv
        all_dat["kv_entry"] = kv_entry
        all_dat["form_en"] = form_en
        all_dat["form_en_entry"] = form_en_entry
        all_dat["surf_en"] = ";".join(map(str, surf_en))
        all_dat["surf_en_entry"] = ";".join(map(str, surf_en_entry))
        all_dat["vac_en"] = ";".join(map(str, vac_en))
        all_dat["vac_en_entry"] = ";".join(map(str, vac_en_entry))

        wad_en, wad_en_entry = self.interface_workflow()

        all_dat["wad_en"] = ";".join(map(str, wad_en))
        all_dat["wad_en_entry"] = ";".join(map(str, wad_en_entry))
        err_wad = ""
        if len(wad_en):
            err_wad = mean_absolute_error(wad_en_entry, wad_en)
        all_dat["timings"] = timings
        error_dat = {}
        if len(initial_a) > 0:
            err_a = mean_absolute_error(initial_a, final_a)
            err_b = mean_absolute_error(initial_b, final_b)
            err_c = mean_absolute_error(initial_c, final_c)
            err_fen = mean_absolute_error(form_en_entry, form_en)
            err_vol = mean_absolute_error(initial_vol, final_vol)
            error_dat["err_a"] = err_a
            error_dat["err_b"] = err_b
            error_dat["err_c"] = err_c
            error_dat["err_form"] = err_fen
            error_dat["err_vol"] = err_vol
        if len(c11) > 0:
            err_c11 = mean_absolute_error(c11_entry, c11)
            err_c44 = mean_absolute_error(c44_entry, c44)
            error_dat["err_c11"] = err_c11
            error_dat["err_c44"] = err_c44
        # err_kv = mean_absolute_error(kv_entry, kv)
        if len(surf_en) > 0:
            err_surf_en = mean_absolute_error(surf_en_entry, surf_en)
            error_dat["err_surf_en"] = err_surf_en
        if len(vac_en) > 0:
            err_vac_en = mean_absolute_error(vac_en_entry, vac_en)
            # error_dat['err_kv']=err_kv
            error_dat["err_vac_en"] = err_vac_en
        error_dat["err_wad"] = err_wad
        error_dat["time"] = np.sum(np.array(timings))
        print("error_dat", error_dat)
        df = pd.DataFrame(all_dat)
        fname = os.path.join(self.output_dir, "dat.csv")
        df.to_csv(fname, index=False)
        print("a", err_a)
        print("b", err_b)
        print("c", err_c)
        print("fen", err_fen)
        print("vol", err_vol)
        print("c11", err_c11)
        print("c44", err_c44)
        # print("kv", err_kv)
        if self.make_plot:
            self.plot_results(fname)
        return (
            error_dat,
            all_dat,
        )

    def plot_results(self, fname="out_mp_tak4_cut4/dat.csv"):
        df = pd.read_csv(fname)
        print(df)

        the_grid = GridSpec(4, 3)
        plt.rcParams.update({"font.size": 18})
        plt.figure(figsize=(16, 14))

        plt.subplot(the_grid[0, 0])
        plt.scatter(df["initial_a"], df["final_a"])
        plt.plot(df["initial_a"], df["initial_a"], c="black", linestyle="-.")
        plt.xlabel("a-DFT ($\AA$)")
        plt.ylabel("a-ML ($\AA$)")
        min_x = min(pd.concat([df["initial_a"], df["final_a"]]))
        max_x = max(pd.concat([df["initial_a"], df["final_a"]]))
        min_y = min(pd.concat([df["initial_a"], df["final_a"]]))
        max_y = max(pd.concat([df["initial_a"], df["final_a"]]))
        # plt.xlim([min_x,max_x])
        # plt.ylim([min_y,max_y])
        title = "(a) " + str(
            round(r2_score(df["initial_a"], df["final_a"]), 2)
        )
        plt.title(title)

        plt.subplot(the_grid[0, 1])
        plt.scatter(df["initial_b"], df["final_b"])
        plt.plot(df["initial_b"], df["initial_b"], c="black", linestyle="-.")
        plt.xlabel("b-DFT ($\AA$)")
        plt.ylabel("b-ML ($\AA$)")
        title = "(b) " + str(
            round(r2_score(df["initial_b"], df["final_b"]), 2)
        )
        plt.title(title)

        plt.subplot(the_grid[0, 2])
        plt.scatter(df["initial_c"], df["final_c"])
        plt.plot(df["initial_c"], df["initial_c"], c="black", linestyle="-.")
        plt.xlabel("c-DFT ($\AA$)")
        plt.ylabel("c-ML ($\AA$)")
        title = "(c) " + str(
            round(r2_score(df["initial_c"], df["final_c"]), 2)
        )
        plt.title(title)

        plt.subplot(the_grid[1, 0])
        plt.scatter(df["form_en_entry"], df["form_en"])
        plt.plot(
            df["form_en_entry"], df["form_en_entry"], c="black", linestyle="-."
        )
        plt.xlabel("$E_f$-DFT (eV/atom)")
        plt.ylabel("$E_f$-ML (eV/atom)")
        title = "(d) " + str(
            round(r2_score(df["form_en_entry"], df["form_en"]), 2)
        )
        plt.title(title)

        plt.subplot(the_grid[1, 1])
        plt.scatter(df["initial_vol"], df["final_vol"])
        plt.plot(
            df["initial_vol"], df["initial_vol"], c="black", linestyle="-."
        )
        plt.xlabel("vol-DFT (${\AA}^3$)")
        plt.ylabel("vol-ML (${\AA}^3$)")
        title = "(e) " + str(
            round(r2_score(df["initial_b"], df["final_b"]), 2)
        )
        plt.title(title)

        plt.subplot(the_grid[1, 2])
        plt.scatter(df["c11_entry"], df["c11"])
        plt.plot(df["c11_entry"], df["c11_entry"], c="black", linestyle="-.")
        plt.xlabel("$C_{11}$-DFT (GPa)")
        plt.ylabel("$C_{11}$-ML (GPa)")
        title = "(f) " + str(round(r2_score(df["c11_entry"], df["c11"]), 2))
        plt.title(title)

        plt.subplot(the_grid[2, 0])
        plt.scatter(df["c44_entry"], df["c44"])
        plt.plot(df["c44_entry"], df["c44_entry"], c="black", linestyle="-.")
        plt.xlabel("$C_{44}$-DFT (GPa)")
        plt.ylabel("$C_{44}$-ML (GPa)")
        title = "(g) " + str(round(r2_score(df["c44_entry"], df["c44"]), 2))
        plt.title(title)

        plt.subplot(the_grid[2, 1])
        x = np.concatenate(
            df["vac_en_entry"].apply(
                lambda x: np.array(x.split(";"), dtype="float")
            )
        )
        y = np.concatenate(
            df["vac_en"].apply(lambda x: np.array(x.split(";"), dtype="float"))
        )
        plt.scatter(x, y)
        plt.plot(x, x, linestyle="-.", c="black")
        title = "(h) " + str(round(r2_score(x, y), 2))
        plt.title(title)
        plt.xlabel("$E_{vac}$-DFT (eV)")
        plt.ylabel("$E_{vac}$-ML (eV)")

        plt.subplot(the_grid[2, 2])
        x = np.concatenate(
            df["surf_en_entry"].apply(
                lambda x: np.array(x.split(";"), dtype="float")
            )
        )
        y = np.concatenate(
            df["surf_en"].apply(
                lambda x: np.array(x.split(";"), dtype="float")
            )
        )
        plt.scatter(x, y)
        plt.plot(x, x, linestyle="-.", c="black")
        title = "(i) " + str(round(r2_score(x, y), 2))
        plt.title(title)
        plt.xlabel("$E_{surf}$-DFT (J/m2)")
        plt.ylabel("$E_{surf}$-ML (J/m2)")

        plt.subplot(the_grid[3, 0])
        x = np.concatenate(
            df["wad_en_entry"].apply(
                lambda x: np.array(x.split(";"), dtype="float")
            )
        )
        y = np.concatenate(
            df["wad_en"].apply(lambda x: np.array(x.split(";"), dtype="float"))
        )

        print("wadx", x)
        print("wady", y)
        plt.scatter(x, y)
        plt.plot(x, x, c="black", linestyle="-.")
        plt.xlabel("$W_{ad}$-prev (J$m^{_2}$)")
        plt.ylabel("$W_{ad}$-ML(J$m^-{2}$)")
        # plt.ylim([min_y,max_y])
        title = "(h) " + str(round(r2_score(x, y), 2))
        plt.title(title)

        pname = fname.replace("dat.csv", "dat.png")
        plt.tight_layout()
        plt.savefig(pname)
        plt.close()

    def elastic_tensor(
        self, atoms=None, id=None, entry=None, entry_key="elastic_tensor"
    ):
        from elastic import get_elementary_deformations, get_elastic_tensor
        import elastic

        ase_atoms = atoms.ase_converter()
        ase_atoms.calc = self.calculator
        systems = get_elementary_deformations(ase_atoms)
        cij_order = elastic.elastic.get_cij_order(ase_atoms)
        Cij, Bij = get_elastic_tensor(ase_atoms, systems)
        c11 = ""
        c44 = ""
        c11_entry = ""
        c44_entry = ""
        info = {}
        for i, j in zip(cij_order, Cij):
            # print(i, j / ase.units.GPa)
            if i == "C_11":
                c11 = j / ase.units.GPa
            if i == "C_44":
                c44 = j / ase.units.GPa
        if entry is not None and entry_key in entry:
            et = entry[entry_key]
            c11_entry = et[0][0]
            c44_entry = et[3][3]
        info["c11"] = c11
        info["c44"] = c44
        info["c11_entry"] = c11_entry
        info["c44_entry"] = c44_entry
        print("C11", c11, c11_entry)
        print("C44", c44, c44_entry)
        # name = id + "_elast.json"
        # fname = os.path.join(self.output_dir, name)
        # dumpjson(data=info, filename=fname)
        return info

    def surface_energy(
        self,
        atoms=None,
        jid="x",
        cell_relax=False,
        id=None,
        entry=None,
        entry_key="surface",
    ):
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
                "Surface-"
                + str(id)
                + "_"
                + str("miller")
                # + str(surf.composition.reduced_formula)
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
                    - bulk_enp * (surf.num_atoms)
                )
                / (2 * area)
            )
            info = {}
            info["name"] = name
            info["surf_en"] = surf_en
            info["surf_en_entry"] = ""
            if (
                entry is not None
                and entry_key in entry
                and name in entry[entry_key]
            ):
                print("pred", name, surf_en, "val", entry[entry_key][name])
                info["surf_en_entry"] = entry[entry_key][name]
            surface_results.append(info)
        # if entry is not None and entry_key in entry:
        #    print('surface',entry_key,entry[entry_key])
        # name = id + "_surf.json"
        # fname = os.path.join(self.output_dir, name)
        # dumpjson(data=info, filename=fname)
        return surface_results

    def vacancy_energy(
        self,
        atoms=None,
        jid="x",
        cell_relax=False,
        id=None,
        entry=None,
        entry_key="vacancy",
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
            print(
                "id",
                id,
                "symbol",
                j.to_dict()["symbol"],
                "wyc",
                j.to_dict()["wyckoff_multiplicity"],
            )
            name = (
                str(id)
                + "_"
                + j.to_dict()["symbol"]
                # Ignoring wyckoff notation
                # + "_"
                # + j.to_dict()["wyckoff_multiplicity"].split("_")[0]
            )

            if j.to_dict()["symbol"] in self.chem_pot_ref:
                atoms_en = self.chem_pot_ref[j.to_dict()["symbol"]]
            else:
                jid_elemental = chem_pot[j.to_dict()["symbol"]]["jid"]
                atoms_elemental = Atoms.from_dict(
                    get_entry(jid_elemental)["atoms"]
                )
                # atoms_elemental = Atoms.from_dict(
                #    get_jid_data(jid=jid_elemental, dataset="dft_3d")["atoms"]
                # )
                atoms_en = (
                    self.get_energy(
                        atoms=atoms_elemental,
                        cell_relax=self.chem_pot_relax,
                        constant_volume=False,
                    )[0]
                ) / atoms_elemental.num_atoms
                self.chem_pot_ref[j.to_dict()["symbol"]] = atoms_en
            defect_en = (
                self.get_energy(
                    atoms=strt, cell_relax=cell_relax, constant_volume=False
                )[0]
                - bulk_enp * (strt.num_atoms + 1)
                + atoms_en
            )
            info = {}
            info["name"] = name
            info["vac_en"] = defect_en
            info["vac_en_entry"] = ""
            print(name, defect_en)
            if (
                entry is not None
                and entry_key in entry
                and name in entry[entry_key]
            ):
                print("pred", name, defect_en, "val", entry[entry_key][name])
                info["vac_en_entry"] = entry[entry_key][name]
            vacancy_results.append(info)
        if entry is not None and entry_key in entry:
            print("vacancy", entry_key, entry[entry_key])
        # name = id + "_vac.json"
        # fname = os.path.join(self.output_dir, name)
        # dumpjson(data=info, filename=fname)
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
        force_mult_natoms=True,
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
            # if force_mult_natoms:
            #    forces*=len(ase_atoms)
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
        nm = id + "_" + atoms.composition.reduced_formula + "_phonon.png"
        phonopy_bands_figname = os.path.join(self.output_dir, nm)
        plt.savefig(phonopy_bands_figname)
        plt.show()
        plt.close()
        # name = id + "_phon.json"
        # fname = os.path.join(self.output_dir, name)
        # dumpjson(data=[zpe], filename=fname)

        return zpe

    def ev_curve(
        self,
        atoms=None,
        dx=np.arange(-0.02, 0.02, 0.001),
        # dx=np.arange(-0.05, 0.05, 0.005),
        id=None,
        entry=None,
        entry_key="bulk_modulus_kv",
    ):
        """Get EV curve."""
        atoms = atoms.get_conventional_atoms.make_supercell([2, 2, 2])
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
        kv = "na"
        try:
            eos = EquationOfState(vol, y, eos="murnaghan")
            v0, e0, B = eos.fit()
            kv = B / kJ * 1.0e24  # , 'GPa')
        except:
            pass
        # print("Energies:", y)
        # print("Volumes:", vol)
        nm = id + "_" + atoms.composition.reduced_formula + "_eV.png"
        plt.plot(vol, y, "-o")
        fname = self.output_dir + "/" + nm
        plt.savefig(fname)
        plt.close()
        # name = id + "_ev.json"
        # fname = os.path.join(self.output_dir, name)
        # dumpjson(data=[kv], filename=fname)
        info = {}
        kv_entry = ""
        if entry is not None:
            kv_entry = entry[entry_key]
        info["kv"] = kv
        info["kv_entry"] = kv_entry
        print("Kv", kv, kv_entry)

        return info

    def interface_workflow(self):
        wad_en = []
        wad_en_entry = []
        df_metals = pd.read_csv(
            "https://figshare.com/ndownloader/files/49065421"
        )
        df_insulators = pd.read_csv(
            "https://figshare.com/ndownloader/files/49065418"
        )
        df = pd.concat([df_insulators, df_metals])
        dataset1 = data("dft_3d")
        dataset2 = data("dft_2d")
        dataset = dataset1 + dataset2

        for i, ii in tqdm(df.iterrows(), total=len(df)):
            if len(wad_en_entry) < 3:
                # if ii['Film'] in metals and ii['Subs'] in metals:
                film_ids = []
                subs_ids = []
                film_indices = []
                subs_indices = []
                # try:
                film_ids.append("JVASP-" + str(ii["JARVISID-Film"]))
                subs_ids.append("JVASP-" + str(ii["JARVISID-Subs"]))
                film_indices.append(
                    [
                        int(ii["Film-miller"][1]),
                        int(ii["Film-miller"][2]),
                        int(ii["Film-miller"][3]),
                    ]
                )
                subs_indices.append(
                    [
                        int(ii["Subs-miller"][1]),
                        int(ii["Subs-miller"][2]),
                        int(ii["Subs-miller"][3]),
                    ]
                )
                print(
                    film_indices[-1],
                    subs_indices[-1],
                    film_ids[-1],
                    subs_ids[-1],
                )
                x = InterfaceCombi(
                    dataset=dataset,
                    film_indices=film_indices,
                    subs_indices=subs_indices,
                    film_ids=film_ids,
                    subs_ids=subs_ids,
                    disp_intvl=0.0,
                )
                wads = x.calculate_wad(
                    method="other",
                    extra_params={"calculator": self.calculator},
                )
                wads = np.array(x.wads["wads"])
                index = np.argmin(wads)
                wad_en.append(wads[index])
                wad_en_entry.append(float(ii["PrevData W_adhesion (Jm-2)"]))
                print(ii)
                print("Pred Wad", -1 * wads[index])
                print()
                # wads = x.calculate_wad(method="vasp", index=index)
                # wads = x.calculate_wad_vasp(sub_job=True)
        # except:
        #  pass
        return wad_en, wad_en_entry


# metal_metal_interface_workflow(calc)


if __name__ == "__main__":
    jids_check = [
        "JVASP-1002",  # Si
        "JVASP-1174",  # GaAs F-43m
        "JVASP-890",  # Ge
        "JVASP-8169",  # GaN F-43m
        "JVASP-8158",  # SiC F-43m
        "JVASP-1372",  # AlAs F-43m
        "JVASP-1186",  # InAs F-43M
        "JVASP-8158",  # SiC F-43m
        "JVASP-7844",  # AlN F-43m
        "JVASP-35106",  # Al3GaN4 P-43m
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
    ]
    print(len(jids_check))
    for i in jids_check:
        print(get_entry(i)["bulk_modulus_kv"])

    atoms_dataset = []
    for i in jids_check:
        print(i)
        entry = get_entry(i)
        atoms = entry["atoms"]
        atoms_dataset.append({"atoms": atoms, "jid": i, "entry": entry})

    # M3GNet
    t1 = time.time()
    ev = Evaluator(
        atoms_dataset=atoms_dataset,
        output_dir="out_m3gnet",
        calculator_type="M3GNet",
    )
    all_dat_matgl, _ = ev.run_all()
    t2 = time.time()
    matgl_time = t2 - t1

    # ALIGNN-FF
    t1 = time.time()
    ev = Evaluator(
        atoms_dataset=atoms_dataset,
        output_dir="out_mp_tak4_cut4",
        calculator_type="ALIGNN-FF",
        alignn_model_path="aff307k_lmdb_param_low_rad_use_force_mult_mp_tak4_cut4/out111",
        # alignn_model_path="aff307k_lmdb_param_low_rad_use_force_mult_mp_tak4/out111c/",
        # alignn_model_path="aff307k_lmdb_param_low_rad_use_cutoff_take4_noforce_mult_cut4/out111a",
    )
    all_dat_mp_tak4_cut4, _ = ev.run_all()
    t2 = time.time()
    al_time = t2 - t1

    # ALIGNN-FF
    t1 = time.time()
    ev = Evaluator(
        atoms_dataset=atoms_dataset,
        output_dir="out_mp_tak4",
        calculator_type="ALIGNN-FF",
        alignn_model_path="aff307k_lmdb_param_low_rad_use_force_mult_mp_tak4/out111c",
        # alignn_model_path="aff307k_lmdb_param_low_rad_use_force_mult_mp_tak4/out111c/",
        # alignn_model_path="aff307k_lmdb_param_low_rad_use_cutoff_take4_noforce_mult_cut4/out111a",
    )
    all_dat_mp_tak4, _ = ev.run_all()
    t2 = time.time()
    al_time = t2 - t1

    # ALIGNN-FF
    t1 = time.time()
    ev = Evaluator(
        atoms_dataset=atoms_dataset,
        output_dir="out_noforce_mult_cut4",
        calculator_type="ALIGNN-FF",
        alignn_model_path="aff307k_lmdb_param_low_rad_use_cutoff_take4_noforce_mult_cut4/out111a",
    )
    all_dat_307_noforce_mult_cut4, _ = ev.run_all()
    t2 = time.time()
    al_time = t2 - t1

    # ALIGNN-FF
    t1 = time.time()
    ev = Evaluator(
        atoms_dataset=atoms_dataset,
        output_dir="out_fd_noforce_mult_cut4",
        calculator_type="ALIGNN-FF",
        alignn_model_path="fd2.5mil_lmdb_param_low_rad_use_cutoff_take4_noforce_mult_cut4/out111/",
    )
    all_dat_fd_noforce_mult_cut4, _ = ev.run_all()
    t2 = time.time()
    al_time = t2 - t1

    # ALIGNN-FF
    t1 = time.time()
    ev = Evaluator(
        atoms_dataset=atoms_dataset,
        output_dir="out_v25",
        calculator_type="ALIGNN-FF",
        alignn_model_path=default_path(),
    )
    v25, _ = ev.run_all()
    t2 = time.time()
    al_time = t2 - t1

    # CHGNET
    t1 = time.time()
    ev = Evaluator(
        atoms_dataset=atoms_dataset,
        output_dir="out_chgnet",
        calculator_type="CHGNet",
    )
    all_dat_chgnet, _ = ev.run_all()
    t2 = time.time()
    chgnet_time = t2 - t1
    df = pd.DataFrame(
        [
            all_dat_matgl,
            all_dat_chgnet,
            all_dat_mp_tak4,
            all_dat_mp_tak4_cut4,
            all_dat_fd_noforce_mult_cut4,
            all_dat_307_noforce_mult_cut4,
            v25,
        ],
        index=["matgl", "chgnet", "mp", "mp4", "fd", "f307", "f527"],
    )

    """
    t1 = time.time()
    ev = Evaluator(
        atoms_dataset=atoms_dataset,
        output_dir="out_chgnet",
        calculator_type="CHGNet",
    )
    all_dat_chgnet, _ = ev.run_all()
    t2 = time.time()
    chgnet_time = t2 - t1

    print("all_dat_v5_27_2024", all_dat_v5_27_2024)
    print("all_dat_matgl", all_dat_matgl)
    print("all_dat_chgnet", all_dat_chgnet)
    """
    print(df)
    import plotly.express as px

    fig = px.imshow(df, text_auto=True)
    # fig.show()
    fig.write_html("error.html")
