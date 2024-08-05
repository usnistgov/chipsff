import alignn
import matgl
import chgnet
import mace
import sevenn

from jarvis.db.figshare import data, get_jid_data
from jarvis.core.atoms import Atoms
from jarvis.io.vasp.inputs import Poscar
indices_list = [[1, 0, 0],[1, 1, 1],[1,1,0],[0,1,1],[0,0,1],[0,1,0]]
def get_atoms(jid):
    dat = get_jid_data(jid=jid, dataset="dft_3d")
    return Atoms.from_dict(dat["atoms"])

from ase.optimize.fire import FIRE
from ase.constraints import ExpCellFilter
from jarvis.core.atoms import ase_to_atoms
from matgl.ext.ase import M3GNetCalculator
from alignn.ff.ff import AlignnAtomwiseCalculator, default_path
from chgnet.model.dynamics import CHGNetCalculator
from sevenn.sevennet_calculator import SevenNetCalculator
model_filename='best_model.pt'
from mace.calculators import mace_mp

def setup_calculator(calculator_type):
    if calculator_type == "matgl":
        matgl_path = "/users/dtw2/matgl/pretrained_models/M3GNet-MP-2021.2.8-PES/"
        pot = matgl.load_model(matgl_path)
        return M3GNetCalculator(pot)
    elif calculator_type == "alignn_ff":
        model_path = default_path()
        return AlignnAtomwiseCalculator(path=model_path,stress_wt=0.3,force_mult_natoms=False,force_multiplier=16,modl_filename=model_filename)
    elif calculator_type == "chgnet":
        return CHGNetCalculator()
    elif calculator_type == "mace":
        return mace_mp()
    elif calculator_type == "sevennet":
        checkpoint_path = "/users/dtw2/SevenNet/pretrained_potentials/SevenNet_0__11July2024/checkpoint_sevennet_0.pth"
        sevenet_cal = SevenNetCalculator(checkpoint_path, device='cpu')
        return sevenet_cal
    else:
        raise ValueError("Unsupported calculator type")

import io
import contextlib
import re
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

import numpy as np
from ase.eos import EquationOfState
from ase.units import kJ
import matplotlib.pyplot as plt

def ev_curve(
    atoms=None,
    dx=np.arange(-0.05, 0.05, 0.01),
    calculator=None,
    on_relaxed_struct=False,
    stress_wt=1,
):
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
    kv = B / kJ * 1.0e24  # , 'GPa')
    print("Energies:", y)
    print("Volumes:", vol)
    return x, y, eos, kv

def calculate_ev_curve(atoms, calculator, dx=np.arange(-0.06, 0.06, 0.01)):
    x, y, eos, kv = ev_curve(atoms=atoms, dx=dx, calculator=calculator)
    v0, e0 = eos.v0, eos.e0  # Equilibrium volume and energy
    return x, y, eos, kv, e0, v0

import numpy as np
from phonopy import Phonopy
from phonopy.file_IO import write_FORCE_CONSTANTS
from jarvis.core.kpoints import Kpoints3D as Kpoints
from ase import Atoms as AseAtoms

def phonons_gen(
    atoms=None,
    calculator=None,
    dim=[2, 2, 2],
    freq_conversion_factor=33.3566830,  # Thz to cm-1
    phonopy_bands_figname="phonopy_bands.png",
    write_fc=False,
    min_freq_tol=-0.05,
    distance=0.2,
):
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
    plt.fill_between(
        ds, freqs, color=(0.2, 0.4, 0.6, 0.6), edgecolor="k", lw=1, y2=0
    )
    plt.xlabel("DOS")
    plt.yticks([])
    plt.xticks([])
    plt.ylim([min_freq, max_freq])
    plt.xlim([0, max(ds)])
    plt.tight_layout()
    plt.savefig(phonopy_bands_figname)
    #plt.show()
    plt.close()

    return phonon



from elastic import get_elementary_deformations, get_elastic_tensor
import elastic
import ase.units

def calculate_elastic_tensor(atoms, calculator):
    ase_atoms = atoms.ase_converter()
    ase_atoms.calc = calculator
    systems = get_elementary_deformations(ase_atoms)
    cij_order = elastic.elastic.get_cij_order(ase_atoms)
    Cij, Bij = get_elastic_tensor(ase_atoms, systems)
    elastic_tensor = {i: j / ase.units.GPa for i, j in zip(cij_order, Cij)}
    return elastic_tensor
    print(elastic_tensor)

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



import numpy as np
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.nvtberendsen import NVTBerendsen
from jarvis.core.atoms import ase_to_atoms, Atoms
from ase.io import write
#from ase.geometry.rdf import get_containing_cell_length, get_rdf
from ase.ga.utilities import get_rdf

def get_supercell_dims(atoms, target_atoms=200):
    n_atoms = len(atoms.elements)
    scale = np.ceil((target_atoms / n_atoms) ** (1/3))
    return [int(scale)] * 3

def ensure_cell_size(ase_atoms, min_size):
    """ Ensure that all cell dimensions are at least min_size """
    cell_lengths = ase_atoms.get_cell().lengths()
    scale_factors = np.ceil(min_size / cell_lengths).astype(int)
    supercell_dims = [max(1, scale) for scale in scale_factors]
    return supercell_dims

def general_melter(jid='', atoms=None, calculator=None, log_file=None, job_info=None, output_dir="", min_size=10.0):
    if atoms is None or calculator is None:
        raise ValueError("Atoms and calculator must be provided")

    ase_atoms = atoms.ase_converter()
    dim = ensure_cell_size(ase_atoms, min_size)
    sup = atoms.make_supercell_matrix(dim)

    ase_atoms = sup.ase_converter()
    ase_atoms.calc = calculator

    dt = 1 * units.fs
    temp0, nsteps0 = 3000, 5000
    temp1, nsteps1 = 300, 5000
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

def calculate_rdf(ase_atoms, jid, calculator_type, output_dir, log_file, job_info, rmax=4.5, nbins=200):
    def perform_rdf_calculation(rmax):
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
        return rdf_plot_filename

    try:
        perform_rdf_calculation(rmax)
    except ValueError as e:
        if "The cell is not large enough" in str(e):
            # Extract the suggested rmax value from the error message
            recommended_rmax = float(str(e).split('<')[1].split('=')[1])
            log_job_info(f'Error: {e}. Adjusting rmax to {recommended_rmax}.', log_file)
            perform_rdf_calculation(recommended_rmax)
        else:
            log_job_info(f'Error: {e}', log_file)
            raise

import json

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

chemical_potentials_json = '''{
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
}'''


# Load chemical potentials from JSON data
chemical_potentials = json.loads(chemical_potentials_json)

# Function to get the chemical potential for a given element
def get_chemical_potential(element, calculator):
    return chemical_potentials[element][f"energy_{calculator}"]

# Function to calculate surface energy
def calculate_surface_energy(surface_energy, bulk_energy, num_units, surface_area):
    return (surface_energy - bulk_energy * num_units) * 16.02176565 / (2 * surface_area)

# Function to calculate vacancy formation energy
def calculate_vacancy_formation_energy(vacancy_energy, bulk_energy, chemical_potential):
    return vacancy_energy - bulk_energy + chemical_potential

def calculate_formation_energy_per_atom(e0, atoms, calculator_type):
    composition = atoms.composition.to_dict()
    total_energy = e0
    for element, amount in composition.items():
        total_energy -= get_chemical_potential(element, calculator_type) * amount
    formation_energy_per_atom = total_energy / atoms.num_atoms
    return formation_energy_per_atom

# Commented out IPython magic to ensure Python compatibility.
import os
import time
import matplotlib.pyplot as plt
from jarvis.analysis.defects.vacancy import Vacancy
from jarvis.analysis.defects.surface import Surface
from jarvis.core.atoms import Atoms, get_supercell_dims
from jarvis.analysis.structure.spacegroup import (
    Spacegroup3D,
    symmetrically_distinct_miller_indices,
)
from alignn.ff.ff import phonons
from alignn.ff.ff import phonons3
model_path = default_path()
#calculator_types = ["matgl","chgnet","alignn_ff","mace"]

ids = ['JVASP-30']
calculator_types = ["alignn_ff","chgnet","mace","sevennet"]

for calculator_type in calculator_types:
    for jid in ids:
        try:
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

            start_time = time.time()
            relaxed_atoms = relax_structure(atoms, calculator_type, log_file=log_file, job_info=job_info)
            log_job_info(f'Relaxed structure for {jid}: {relaxed_atoms.to_dict()}', log_file)
            job_info["relaxed_atoms"] = relaxed_atoms.to_dict()
            end_time = time.time()
            log_job_info(f"Relaxation time: {end_time - start_time} seconds", log_file)
            

            start_time = time.time()
            quenched_atoms = general_melter(jid=jid, atoms=relaxed_atoms, calculator=calculator, log_file=log_file, job_info=job_info, output_dir=output_dir)
            calculate_rdf(quenched_atoms.ase_converter(), jid, calculator_type, output_dir, log_file, job_info)
            save_dict_to_json(job_info, job_info_filename)
            end_time = time.time()
            log_job_info(f"Melt/Quench time: {end_time - start_time} seconds", log_file)
            condensed_results = {
                "calculator_type": calculator_type,
                "jid": jid,
            }

            condensed_results_filename = os.path.join(output_dir, f"{jid}_{calculator_type}_final_results.json")
            save_dict_to_json(condensed_results, condensed_results_filename)
            log_job_info(f"Final results saved to {condensed_results_filename}", log_file)

        except Exception as e:
            log_job_info(f"Error processing {jid} with calculator {calculator_type}: {e}", log_file)

