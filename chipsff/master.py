import alignn
import matgl
import chgnet
import mace
import numpy as np
import matplotlib.pyplot as plt
import io
import contextlib
import re
import h5py
import shutil
import glob
import json
import subprocess
import os
from ase.optimize.fire import FIRE
from ase.constraints import ExpCellFilter
from ase.units import kJ, units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.nvtberendsen import NVTBerendsen
from ase.io import write
from ase import Atoms as AseAtoms
from ase.geometry.rdf import CellTooSmall
from ase.eos import EquationOfState
from jarvis.db.figshare import data, get_jid_data
from jarvis.core.atoms import Atoms, ase_to_atoms, get_supercell_dims
from jarvis.io.vasp.inputs import Poscar
from jarvis.core.kpoints import Kpoints3D as Kpoints
from jarvis.analysis.defects.vacancy import Vacancy
from jarvis.analysis.defects.surface import Surface
from jarvis.db.jsonutils import dumpjson, loadjson
from matgl.ext.ase import M3GNetCalculator
from alignn.ff.ff import AlignnAtomwiseCalculator, default_path
from chgnet.model.dynamics import CHGNetCalculator
from mace.calculators import mace_mp
from phonopy import Phonopy
from phonopy.file_IO import write_FORCE_CONSTANTS

# Constants
model_filename = 'best_model.pt'
indices_list = [[1, 0, 0], [1, 1, 1], [1, 1, 0], [0, 1, 1], [0, 0, 1], [0, 1, 0]]
jids_check = [
    "JVASP-1002", "JVASP-816", "JVASP-867", "JVASP-1029", "JVASP-861", "JVASP-30",
    "JVASP-8169", "JVASP-890", "JVASP-8158", "JVASP-8118", "JVASP-107", "JVASP-39",
    "JVASP-7844", "JVASP-35106", "JVASP-1174", "JVASP-1372", "JVASP-91", "JVASP-1186",
    "JVASP-1408", "JVASP-105410", "JVASP-1177", "JVASP-79204", "JVASP-1393", "JVASP-1312",
    "JVASP-1327", "JVASP-1183", "JVASP-1192", "JVASP-8003", "JVASP-96", "JVASP-1198",
    "JVASP-1195", "JVASP-9147", "JVASP-41", "JVASP-34674", "JVASP-113", "JVASP-32"
]
model_path = default_path()

# Helper Functions
def get_atoms(jid):
    dat = get_jid_data(jid=jid, dataset="dft_3d")
    return Atoms.from_dict(dat["atoms"])

def setup_calculator(calculator_type):
    if calculator_type == "matgl":
        pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        return M3GNetCalculator(pot)
    elif calculator_type == "alignn_ff":
        model_path = default_path()
        return AlignnAtomwiseCalculator(path=model_path, stress_wt=0.3, force_mult_natoms=False, force_multiplier=1, modl_filename=model_filename)
    elif calculator_type == "chgnet":
        return CHGNetCalculator()
    elif calculator_type == "mace":
        return mace_mp()
    else:
        raise ValueError("Unsupported calculator type")

def capture_fire_output(ase_atoms, fmax, steps):
    log_stream = io.StringIO()
    with contextlib.redirect_stdout(log_stream):
        dyn = FIRE(ase_atoms)
        dyn.run(fmax=fmax, steps=steps)
    output = log_stream.getvalue().strip()
    final_energy = None
    if output:
        last_line = output.split('\n')[-1]
        match = re.search(r'FIRE:\s+\d+\s+\d+:\d+:\d+\s+(-?\d+\.\d+)', last_line)
        if match:
            final_energy = float(match.group(1))
    return final_energy, dyn.nsteps

def relax_structure(atoms, calculator_type, fmax=0.05, steps=200, log_file=None, job_info=None):
    ase_atoms = atoms.ase_converter()
    calc = setup_calculator(calculator_type)
    ase_atoms.calc = calc
    ase_atoms = ExpCellFilter(ase_atoms)
    final_energy, nsteps = capture_fire_output(ase_atoms, fmax, steps)
    relaxed_atoms = ase_to_atoms(ase_atoms.atoms)
    converged = nsteps < steps
    if log_file:
        log_job_info(f"Final energy of FIRE optimization for structure: {final_energy}", log_file)
        log_job_info(f"Relaxation {'converged' if converged else 'did not converge'} within {steps} steps.", log_file)
    if job_info is not None:
        job_info["final_energy_structure"] = final_energy
        job_info["converged"] = converged
    return relaxed_atoms

def ev_curve(atoms=None, dx=np.arange(-0.05, 0.05, 0.01), calculator=None, on_relaxed_struct=False, stress_wt=1):
    """Get EV curve."""
    relaxed = atoms
    y = []
    vol = []
    for i in dx:
        s1 = relaxed.strain_atoms(i)
        ase_atoms = s1.ase_converter()
        ase_atoms.calc = calculator
        energy = ase_atoms.get_potential_energy()
        y.append(energy)
        vol.append(s1.volume)
    x = np.array(dx)
    y = np.array(y)
    eos = EquationOfState(vol, y, eos="murnaghan")
    v0, e0, B = eos.fit()
    kv = B / kJ * 1.0e24  # GPa
    print("Energies:", y)
    print("Volumes:", vol)
    return x, y, eos, kv

def calculate_ev_curve(atoms, calculator, dx=np.arange(-0.06, 0.06, 0.01)):
    x, y, eos, kv = ev_curve(atoms=atoms, dx=dx, calculator=calculator)
    v0, e0 = eos.v0, eos.e0  # Equilibrium volume and energy
    return x, y, eos, kv, e0, v0

def phonons_gen(atoms=None, calculator=None, dim=[2, 2, 2], freq_conversion_factor=33.3566830, phonopy_bands_figname="phonopy_bands.png", write_fc=False, min_freq_tol=-0.05, distance=0.2):
    """Make Phonon calculation setup."""
    if atoms is None or calculator is None:
        raise ValueError("Atoms and calculator must be provided")
    kpoints = Kpoints().kpath(atoms, line_density=5)
    bulk = atoms.phonopy_converter()
    phonon = Phonopy(bulk, [[dim[0], 0, 0], [0, dim[1], 0], [0, 0, dim[2]]])
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
        ase_atoms.calc = calculator
        forces = np.array(ase_atoms.get_forces())
        drift_force = forces.sum(axis=0)
        for force in forces:
            force -= drift_force / forces.shape[0]
        set_of_forces.append(forces)
    phonon.produce_force_constants(forces=set_of_forces)
    if write_fc:
        write_FORCE_CONSTANTS(phonon.get_force_constants(), filename="FORCE_CONSTANTS")
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
    plt.ylabel("Frequency (cm$^{-1}$)")
    plt.xlim([0, max(lbls_x)])
    phonon.run_mesh([40, 40, 40], is_gamma_center=True, is_mesh_symmetry=False)
    phonon.run_total_dos()
    tdos = phonon._total_dos
    freqs, ds = tdos.get_dos()
    freqs = np.array(freqs)
    freqs = freqs * freq_conversion_factor
    min_freq = min_freq_tol * freq_conversion_factor
    max_freq = max(freqs)
    plt.ylim([min_freq, max_freq])
    plt.subplot(the_grid[1])
    plt.fill_between(ds, freqs, color=(0.2, 0.4, 0.6, 0.6), edgecolor="k", lw=1, y2=0)
    plt.xlabel("DOS")
    plt.yticks([])
    plt.xticks([])
    plt.ylim([min_freq, max_freq])
    plt.xlim([0, max(ds)])
    plt.tight_layout()
    plt.savefig(phonopy_bands_figname)
    plt.close()
    return phonon

def phonons3_gen(atoms=None, calculator=None, dim=[2, 2, 2], distance=0.2, force_multiplier=2, output_dir=""):
    """Make Phonon3 calculation setup."""
    from phono3py import Phono3py
    ase_atoms = atoms.ase_converter()
    ase_atoms.calc = calculator
    bulk = atoms.phonopy_converter()
    phonon = Phono3py(bulk, [[dim[0], 0, 0], [0, dim[1], 0], [0, 0, dim[2]]])
    phonon.generate_displacements(distance=distance)
    supercells = phonon.supercells_with_displacements
    set_of_forces = []
    for scell in supercells:
        ase_atoms = AseAtoms(
            symbols=scell.get_chemical_symbols(),
            scaled_positions=scell.get_scaled_positions(),
            cell=scell.get_cell(),
            pbc=True,
        )
        ase_atoms.calc = calculator
        forces = np.array(ase_atoms.get_forces())
        drift_force = forces.sum(axis=0)
        for force in forces:
            force -= drift_force / forces.shape[0]
        set_of_forces.append(forces)
    forces = np.array(set_of_forces).reshape(-1, len(phonon.supercell), 3)
    phonon.forces = forces
    phonon.produce_fc3()
    phonon.mesh_numbers = 30
    phonon.init_phph_interaction()
    phonon.run_thermal_conductivity(temperatures=range(0, 1001, 10), write_kappa=True)
    kappa = phonon.thermal_conductivity.kappa
    print(f"Thermal conductivity: {kappa}")
    hdf5_file_pattern = "kappa-*.hdf5"
    for hdf5_file in glob.glob(hdf5_file_pattern):
        shutil.move(hdf5_file, os.path.join(output_dir, hdf5_file))
    return kappa

def convert_kappa_units(hdf5_filename, temperature_index):
    with h5py.File(hdf5_filename, 'r') as f:
        kappa_unit_conversion = f['kappa_unit_conversion'][()]
        heat_capacity = f['heat_capacity'][:]
        gv_by_gv = f['gv_by_gv'][:]
        gamma = f['gamma'][:]
        converted_kappa = kappa_unit_conversion * heat_capacity[temperature_index, 2, 0] * gv_by_gv[2, 0] / (2 * gamma[temperature_index, 2, 0])
        return converted_kappa

def process_phonon3_results(output_dir, log_file, job_info):
    file_pattern = os.path.join(output_dir, "kappa-*.hdf5")
    file_list = glob.glob(file_pattern)
    temperatures = np.arange(10, 101, 10)
    kappa_xx_values = []
    if file_list:
        hdf5_filename = file_list[0]
        log_job_info(f"Processing file: {hdf5_filename}", log_file)
        for temperature_index in temperatures:
            converted_kappa = convert_kappa_units(hdf5_filename, temperature_index)
            kappa_xx = converted_kappa[0]
            kappa_xx_values.append(kappa_xx)
            log_job_info(f"Temperature index {temperature_index}, converted kappa: {kappa_xx}", log_file)
        job_info['temperatures'] = temperatures.tolist()
        job_info['kappa_xx_values'] = kappa_xx_values
        plt.figure(figsize=(8, 6))
        plt.plot(temperatures * 10, kappa_xx_values, marker='o', linestyle='-', color='b')
        plt.xlabel('Temperature (K)')
        plt.ylabel('Converted Kappa (xx element)')
        plt.title('Temperature vs. Converted Kappa (xx element)')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'Temperature_vs_Converted_Kappa.png'))
        plt.close()
    else:
        log_job_info("No files matching the pattern were found.", log_file)

def move_hdf5_files(output_dir):
    file_pattern = "kappa-*.hdf5"
    file_list = glob.glob(file_pattern)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for file in file_list:
        shutil.move(file, os.path.join(output_dir, os.path.basename(file)))

def relax_defect_structure(atoms, calculator_type, fmax=0.05, steps=200, log_file=None, job_info=None):
    ase_atoms = atoms.ase_converter()
    calculator = setup_calculator(calculator_type)
    ase_atoms.calc = calculator
    ase_atoms = ExpCellFilter(ase_atoms, constant_volume=True)
    final_energy, nsteps = capture_fire_output(ase_atoms, fmax, steps)
    relaxed_atoms = ase_to_atoms(ase_atoms.atoms)
    converged = nsteps < steps
    if log_file:
        log_job_info(f"Final energy of FIRE optimization for defect structure: {final_energy}", log_file)
        log_job_info(f"Defect relaxation {'converged' if converged else 'did not converge'} within {steps} steps.", log_file)
    if job_info is not None:
        job_info[f"final_energy_defect for {name}"] = final_energy
        job_info[f"converged for {name}"] = converged
    return relaxed_atoms

def relax_surface_structure(atoms, calculator_type, fmax=0.05, steps=200, log_file=None, job_info=None):
    ase_atoms = atoms.ase_converter()
    calculator = setup_calculator(calculator_type)
    ase_atoms.calc = calculator
    ase_atoms = ExpCellFilter(ase_atoms, constant_volume=True)
    final_energy, nsteps = capture_fire_output(ase_atoms, fmax, steps)
    relaxed_atoms = ase_to_atoms(ase_atoms.atoms)
    converged = nsteps < steps
    if log_file:
        log_job_info(f"Final energy of FIRE optimization for surface structure: {final_energy}", log_file)
        log_job_info(f"Surface relaxation {'converged' if converged else 'did not converge'} within {steps} steps.", log_file)
    if job_info is not None:
        job_info[f"final_energy_surface for {jid} with indices {indices}"] = final_energy
        job_info[f"converged for {jid} with indices {indices}"] = converged
    return relaxed_atoms

def perform_interface_scan(film_jid, substrate_jid, film_index, substrate_index, disp_intvl, calculator_method, output_dir, log_file):
    config = {
        "film_jid": film_jid,
        "substrate_jid": substrate_jid,
        "film_index": film_index,
        "substrate_index": substrate_index,
        "disp_intvl": disp_intvl,
        "calculator_method": calculator_method,
    }
    config_filename = os.path.join(output_dir, f"config_{film_jid}_{substrate_jid}_{calculator_method}.json")
    dumpjson(data=config, filename=config_filename)
    log_job_info(f"Config file created: {config_filename}", log_file)
    command = f"run_intermat.py --config_file {config_filename}"
    log_job_info(f"Running command: {command}", log_file)
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        log_job_info(f"Command output: {result.stdout}", log_file)
    except subprocess.CalledProcessError as e:
        log_job_info(f"Command failed with error: {e.stderr}", log_file)
        return None, None
    main_results_filename = "intermat_results.json"
    output_results_filename = os.path.join(output_dir, main_results_filename)
    if not os.path.exists(main_results_filename):
        log_job_info(f"Results file not found: {main_results_filename}", log_file)
        return None, None
    os.rename(main_results_filename, output_results_filename)
    res = loadjson(output_results_filename)
    w_adhesion = res.get("wads", [])
    systems_info = res.get("systems", {})
    if "wads" in res:
        plt.contourf(res["wads"], cmap="plasma")
        plt.axis("off")
        plot_filename = os.path.join(output_dir, f"Interface_Scan_{film_jid}_{substrate_jid}_{calculator_method}.png")
        plt.savefig(plot_filename)
        plt.show()
        return output_results_filename, plot_filename, w_adhesion, systems_info
    else:
        log_job_info(f"No 'wads' key in results file: {output_results_filename}", log_file)
        return output_results_filename, None, w_adhesion, systems_info

def get_supercell_dims(atoms, target_atoms=200):
    n_atoms = len(atoms.elements)
    scale = np.ceil((target_atoms / n_atoms) ** (1/3))
    return [int(scale)] * 3

def general_melter(jid='', atoms=None, calculator=None, log_file=None, job_info=None, output_dir=""):
    if atoms is None or calculator is None:
        raise ValueError("Atoms and calculator must be provided")
    dim = get_supercell_dims(atoms)
    sup = atoms.make_supercell_matrix(dim)
    ase_atoms = sup.ase_converter()
    ase_atoms.calc = calculator
    dt = 1 * units.fs
    temp0, nsteps0 = 2000, 20  # 2000
    temp1, nsteps1 = 300, 20  # 2000
    taut = 20 * units.fs
    trj = os.path.join(output_dir, f'{jid}_melt.traj')
    MaxwellBoltzmannDistribution(ase_atoms, temp0 * units.kB)
    dyn = NVTBerendsen(ase_atoms, dt, temp0, taut=taut, trajectory=trj)
    def myprint():
        print(f'time={dyn.get_time() / units.fs: 5.0f} fs ' +
              f'T={ase_atoms.get_temperature(): 3.0f} K')
    dyn.attach(myprint, interval=20)
    dyn.run(nsteps0)
    dyn.set_temperature(temp1)
    dyn.run(nsteps1)
    final_atoms = ase_to_atoms(ase_atoms)
    poscar_filename = os.path.join(output_dir, f'POSCAR_{jid}_quenched_{calculator_type}.vasp')
    write(poscar_filename, final_atoms.ase_converter(), format='vasp')
    log_job_info(f'MD simulation completed. Final structure saved to {poscar_filename}', log_file)
    job_info["quenched_atoms"] = final_atoms.to_dict()
    return final_atoms

def calculate_rdf(ase_atoms, jid, calculator_type, output_dir, log_file, job_info, rmax=5.0, nbins=200):
    min_lattice_constant = min(ase_atoms.get_cell().lengths())
    if rmax >= min_lattice_constant / 2:
        rmax = min_lattice_constant / 2
        log_job_info(f"Adjusted rmax to {rmax} based on the minimum lattice constant of the structure.", log_file)
    try:
        rdfs, distances = get_rdf(ase_atoms, rmax, nbins)
        plt.figure()
        plt.plot(distances, rdfs)
        plt.xlabel('Distance (Å)')
        plt.ylabel('RDF')
        plt.title(f'Radial Distribution Function for {jid} ({calculator_type})')
        rdf_plot_filename = os.path.join(output_dir, f'RDF_{jid}_{calculator_type}.png')
        plt.savefig(rdf_plot_filename)
        plt.show()
        job_info["rdf_plot"] = rdf_plot_filename
        log_job_info(f'RDF plot saved to {rdf_plot_filename}', log_file)
    except CellTooSmall as e:
        recommended_rmax = float(str(e).split('=')[-1].split()[0])
        log_job_info(f'CellTooSmall error: {e}. Adjusting rmax to {recommended_rmax}.', log_file)
        rdfs, distances = get_rdf(ase_atoms, recommended_rmax, nbins)
        plt.figure()
        plt.plot(distances, rdfs)
        plt.xlabel('Distance (Å)')
        plt.ylabel('RDF')
        plt.title(f'Radial Distribution Function for {jid} ({calculator_type}) with adjusted rmax')
        rdf_plot_filename = os.path.join(output_dir, f'RDF_{jid}_{calculator_type}_adjusted.png')
        plt.savefig(rdf_plot_filename)
        plt.show()
        job_info["rdf_plot_adjusted"] = rdf_plot_filename
        log_job_info(f'Adjusted RDF plot saved to {rdf_plot_filename}', log_file)

def save_dict_to_json(data_dict, filename):
    with open(filename, 'w') as f:
        json.dump(data_dict, f, indent=4)

def load_dict_from_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def log_job_info(message, log_file):
    with open(log_file, 'a') as f:
        f.write(message + "\n")
    print(message)

# Main Workflow
ids = ['JVASP-1174']
calculator_types = ["mace"]


for calculator_type in calculator_types:
    for jid in ids:
        output_dir = f"{jid}_{calculator_type}"
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, f"{jid}_{calculator_type}_job_log.txt")

        log_job_info(f"Processing with calculator: {calculator_type}", log_file)

        calculator = setup_calculator(calculator_type)
        job_info = {"jid": jid, "calculator_type": calculator_type}

        atoms = get_atoms(jid)
        log_job_info(f'Initial atoms for {jid}: {atoms.to_dict()}', log_file)
        job_info["initial_atoms"] = atoms.to_dict()

        job_info_filename = os.path.join(output_dir, f"{jid}_{calculator_type}_job_info.json")
        save_dict_to_json(job_info, job_info_filename)

        relaxed_atoms = relax_structure(atoms, calculator_type, log_file=log_file, job_info=job_info)
        log_job_info(f'Relaxed structure for {jid}: {relaxed_atoms.to_dict()}', log_file)
        job_info["relaxed_atoms"] = relaxed_atoms.to_dict()

        save_dict_to_json(job_info, job_info_filename)

        poscar = Poscar(atoms=relaxed_atoms)
        poscar_filename = os.path.join(output_dir, f'POSCAR_{jid}_relaxed_{calculator_type}.vasp')
        poscar.write_file(poscar_filename)
        job_info["poscar_relaxed"] = poscar_filename

        x, y, eos, kv, e0, v0 = calculate_ev_curve(relaxed_atoms, calculator)
        log_job_info(f"Bulk modulus for {jid}: {kv} GPa", log_file)
        log_job_info(f"Equilibrium energy for {jid}: {e0} eV", log_file)
        log_job_info(f"Equilibrium volume for {jid}: {v0} Å³", log_file)
        job_info["bulk_modulus"] = kv
        job_info["equilibrium_energy"] = e0
        job_info["equilibrium_volume"] = v0
        job_info["eos_data"] = {"x": x.tolist(), "y": y.tolist()}

        fig = plt.figure()
        eos.plot()
        eos_plot_filename = os.path.join(output_dir, f"E_vs_V_{jid}_{calculator_type}.png")
        fig.savefig(eos_plot_filename)
        plt.close(fig)
        log_job_info(f"EV curve plot saved to {eos_plot_filename}", log_file)
        job_info["eos_plot"] = eos_plot_filename

        save_dict_to_json(job_info, job_info_filename)
        phonopy_bands_figname = os.path.join(output_dir, f"ph_{jid}_{calculator_type}.png")
        if calculator_type == "alignn_ff":
            phonon = phonons(model_path=model_path, atoms=relaxed_atoms, phonopy_bands_figname=phonopy_bands_figname, force_mult_natoms=True)
        else:
            phonon = phonons_gen(atoms=relaxed_atoms, phonopy_bands_figname=phonopy_bands_figname, calculator=setup_calculator(calculator_type))

        plt.figure()
        plt.axis('off')
        plt.imshow(plt.imread(phonopy_bands_figname))
        plt.show()
        plt.close()
        job_info["phonopy_bands"] = phonopy_bands_figname

        save_dict_to_json(job_info, job_info_filename)

        phonon.run_mesh(mesh=[20, 20, 20])
        phonon.run_thermal_properties(t_step=10, t_max=1000, t_min=0)
        tprop_dict = phonon.get_thermal_properties_dict()

        plt.figure()
        plt.plot(tprop_dict['temperatures'], tprop_dict['free_energy'], label='Free energy (kJ/mol)', color='red')
        plt.plot(tprop_dict['temperatures'], tprop_dict['entropy'], label='Entropy (J/K*mol)', color='blue')
        plt.plot(tprop_dict['temperatures'], tprop_dict['heat_capacity'], label='Heat capacity (J/K*mol)', color='green')
        plt.legend()
        plt.xlabel('Temperature (K)')
        plt.ylabel('Thermal Properties')
        plt.title(f'Thermal Properties for {jid} with {calculator_type}')
        thermal_props_plot_filename = os.path.join(output_dir, f"Thermal_Properties_{jid}_{calculator_type}.png")
        plt.savefig(thermal_props_plot_filename)
        plt.show()
        job_info["thermal_properties_plot"] = thermal_props_plot_filename

        zpe = tprop_dict['free_energy'][0] * 0.0103643  # converting from kJ/mol to eV
        log_job_info(f"Zero-point energy for {jid} with {calculator_type}: {zpe} eV", log_file)
        job_info["zero_point_energy"] = zpe

        save_dict_to_json(job_info, job_info_filename)
        if calculator_type == "alignn_ff":
            kappa = phonons3(model_path=model_path, atoms=relaxed_atoms)
        else:
            kappa = phonons3_gen(atoms=relaxed_atoms, calculator=calculator, output_dir=output_dir)

        move_hdf5_files(output_dir)
        process_phonon3_results(output_dir, log_file, job_info)
        save_dict_to_json(job_info, job_info_filename)

        strts = Vacancy(atoms).generate_defects(on_conventional_cell=True, enforce_c_size=8, extend=1)
        for j in strts:
            strt = Atoms.from_dict(j.to_dict()['defect_structure'])
            name = f"{jid}_{strt.composition.reduced_formula}_{j.to_dict()['symbol']}_{j.to_dict()['wyckoff_multiplicity']}"
            log_job_info(f'Defect structure: {name}', log_file)
            pos = Poscar(strt)
            pos.write_file(os.path.join(output_dir, f"POSCAR_{name}.vasp"))
            relaxed_defect_atoms = relax_defect_structure(strt, calculator_type, log_file=log_file, job_info=job_info)
            log_job_info(f"Relaxed defect structure for {name}: {relaxed_defect_atoms.to_dict()}", log_file)
            job_info[f"Relaxed defect structure for {name}"] = relaxed_defect_atoms.to_dict()
            save_dict_to_json(job_info, job_info_filename)
            pos_relaxed = Poscar(relaxed_defect_atoms)
            pos_relaxed.write_file(os.path.join(output_dir, f"POSCAR_{name}_relaxed_{calculator_type}.vasp"))

        for indices in indices_list:
            surface = Surface(atoms=atoms, indices=indices, layers=4, vacuum=18).make_surface().center_around_origin()
            if surface.check_polar:
                log_job_info(f"Skipping polar surface for {jid} with indices {indices}", log_file)
                continue
            poscar_surface = Poscar(atoms=surface)
            poscar_surface.write_file(os.path.join(output_dir, f"POSCAR_{jid}_surface_{indices}_{calculator_type}.vasp"))
            relaxed_surface_atoms = relax_surface_structure(surface, calculator_type, log_file=log_file, job_info=job_info)
            log_job_info(f"Relaxed surface structure for {jid} with indices {indices}: {relaxed_surface_atoms.to_dict()}", log_file)
            job_info[f"Relaxed surface structure for {jid} with indices {indices}"] = relaxed_surface_atoms.to_dict()
            save_dict_to_json(job_info, job_info_filename)
            pos_relaxed_surface = Poscar(relaxed_surface_atoms)
            pos_relaxed_surface.write_file(os.path.join(output_dir, f"POSCAR_{jid}_surface_{indices}_{calculator_type}_relaxed.vasp"))

        save_dict_to_json(job_info, job_info_filename)

        interface_scan_results, interface_scan_plot, w_adhesion, systems_info = perform_interface_scan(
            film_jid=jid,
            substrate_jid=jid,
            film_index="1_1_0",
            substrate_index="1_1_0",
            disp_intvl=0.05,
            calculator_method=calculator_type.lower(),
            output_dir=output_dir,
            log_file=log_file
        )
        if interface_scan_results:
            job_info["interface_scan_results"] = interface_scan_results
            log_job_info(f"Interface scan results saved to {interface_scan_results}", log_file)
        if interface_scan_plot:
            job_info["interface_scan_plot"] = interface_scan_plot
            log_job_info(f"Interface scan plot saved to {interface_scan_plot}", log_file)
        if w_adhesion:
            job_info["w_adhesion"] = w_adhesion
            log_job_info(f"w_adhesion: {w_adhesion}", log_file)
        if systems_info:
            job_info["systems_info"] = systems_info
            log_job_info(f"systems_info: {systems_info}", log_file)
        save_dict_to_json(job_info, job_info_filename)

        quenched_atoms = general_melter(jid=jid, atoms=relaxed_atoms, calculator=calculator, log_file=log_file, job_info=job_info, output_dir=output_dir)
        calculate_rdf(quenched_atoms.ase_converter(), jid, calculator_type, output_dir, log_file, job_info)
        save_dict_to_json(job_info, job_info_filename)
