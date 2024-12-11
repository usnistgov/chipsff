import pytest
import os
from chipsff.run_chipsff import MaterialsAnalyzer
from chipsff.config import CHIPSFFConfig
from chipsff.utils import (
    collect_data,
    get_vacancy_energy_entry,
    get_surface_energy_entry,
    save_dict_to_json,
    load_dict_from_json,
    get_jid_list,
)
from chipsff.calcs import setup_calculator
from chipsff.general_material_analyzer import MaterialsAnalyzer


def test_chipsff_jid_list():
    jids = get_jid_list()
    assert len(jids) >= 104


def test_materials_analyzer_initialization():
    # Define the test path for chemical potentials
    config = CHIPSFFConfig(
        jid="JVASP-1002",
        calculator_type="chgnet",
        properties_to_calculate=["relax_structure"],
        chemical_potentials_file="../chipsff/chemical_potentials.json",  # Replace with actual path
    )
    analyzer = MaterialsAnalyzer(
        jid=config.jid,
        calculator_type=config.calculator_type,
        properties_to_calculate=config.properties_to_calculate,
        chemical_potentials_file=config.chemical_potentials_file,
    )
    assert analyzer.jid == "JVASP-1002"
    assert analyzer.calculator_type == "chgnet"


def test_relax_structure():
    # Define the test path for chemical potentials
    config = CHIPSFFConfig(
        jid="JVASP-1002",
        calculator_type="chgnet",
        properties_to_calculate=["relax_structure"],
        chemical_potentials_file="../chipsff/chemical_potentials.json",  # Replace with actual path
    )
    analyzer = MaterialsAnalyzer(
        jid=config.jid,
        calculator_type=config.calculator_type,
        properties_to_calculate=config.properties_to_calculate,
        chemical_potentials_file=config.chemical_potentials_file,
    )
    relaxed_atoms = analyzer.relax_structure()
    assert relaxed_atoms is not None
    assert hasattr(relaxed_atoms, "lattice")


# def test_get_entry():
#    jid = 'JVASP-1002'  # Replace with a valid JID from your dataset
#    entry = get_entry(jid)
#    assert entry['jid'] == jid


def test_collect_data():
    aggregated_data = collect_data()
    assert isinstance(aggregated_data, list)
    assert len(aggregated_data) > 0
    for entry in aggregated_data:
        assert "vacancy" in entry
        assert "surface" in entry


def test_get_vacancy_energy_entry():
    aggregated_data = collect_data()
    jid = "JVASP-1002"  # Use a JID known to have vacancy data
    vacancy_entries = get_vacancy_energy_entry(jid, aggregated_data)
    assert isinstance(vacancy_entries, list)
    for entry in vacancy_entries:
        assert "symbol" in entry
        assert "vac_en_entry" in entry


def test_get_surface_energy_entry():
    aggregated_data = collect_data()
    jid = "JVASP-1002"  # Use a JID known to have surface data
    surface_entries = get_surface_energy_entry(jid, aggregated_data)
    assert isinstance(surface_entries, list)
    for entry in surface_entries:
        assert "name" in entry
        assert "surf_en_entry" in entry


def test_calculate_ev_curve():
    analyzer = MaterialsAnalyzer(
        jid="JVASP-1002",
        calculator_type="chgnet",
        properties_to_calculate=["relax_structure", "calculate_ev_curve"],
        chemical_potentials_file="../chipsff/chemical_potentials.json",
    )
    relaxed_atoms = analyzer.relax_structure()
    result = analyzer.calculate_ev_curve(relaxed_atoms)
    assert result is not None
    vol, y, strained_structures, eos, kv, e0, v0 = result
    assert kv is not None
    assert isinstance(kv, float)


def test_calculate_elastic_tensor():
    analyzer = MaterialsAnalyzer(
        jid="JVASP-1002",
        calculator_type="chgnet",
        properties_to_calculate=[
            "relax_structure",
            "calculate_elastic_tensor",
        ],
        chemical_potentials_file="../chipsff/chemical_potentials.json",
    )
    relaxed_atoms = analyzer.relax_structure()
    elastic_tensor = analyzer.calculate_elastic_tensor(relaxed_atoms)
    assert elastic_tensor is not None
    assert "C_11" in elastic_tensor
    assert isinstance(elastic_tensor["C_11"], float)


def test_run_phonon_analysis():
    analyzer = MaterialsAnalyzer(
        jid="JVASP-1002",
        calculator_type="chgnet",
        properties_to_calculate=["relax_structure", "run_phonon_analysis"],
        chemical_potentials_file="../chipsff/chemical_potentials.json",
    )
    relaxed_atoms = analyzer.relax_structure()
    phonon, zpe = analyzer.run_phonon_analysis(relaxed_atoms)
    assert phonon is not None
    assert zpe is not None
    assert isinstance(zpe, float)


def test_load_chemical_potentials():
    analyzer = MaterialsAnalyzer(
        jid="JVASP-1002",
        calculator_type="chgnet",
        chemical_potentials_file="../chipsff/chemical_potentials.json",
    )
    chemical_potentials = analyzer.load_chemical_potentials()
    assert isinstance(chemical_potentials, dict)


def test_calculate_element_chemical_potential():
    analyzer = MaterialsAnalyzer(
        jid="JVASP-1002",
        calculator_type="chgnet",
        chemical_potentials_file="../chipsff/chemical_potentials.json",
    )
    element = "Si"
    element_jid = "JVASP-1002"
    chem_pot = analyzer.calculate_element_chemical_potential(
        element, element_jid
    )
    assert isinstance(chem_pot, float)


def test_capture_fire_output():
    analyzer = MaterialsAnalyzer(
        jid="JVASP-1002",
        calculator_type="chgnet",
        chemical_potentials_file="../chipsff/chemical_potentials.json",
    )
    ase_atoms = analyzer.atoms.ase_converter()
    ase_atoms.calc = analyzer.calculator
    fmax = 0.05
    steps = 5  # Use small number for testing
    final_energy, nsteps = analyzer.capture_fire_output(
        ase_atoms, fmax, steps, verbose=False
    )
    assert final_energy is not None
    assert nsteps <= steps


def test_save_and_load_dict_json():
    test_dict = {"a": 1, "b": 2}
    test_filename = "test_json_file.json"
    save_dict_to_json(test_dict, test_filename)
    loaded_dict = load_dict_from_json(test_filename)
    assert test_dict == loaded_dict
    os.remove(test_filename)  # Clean up test file


def test_calculate_formation_energy():
    analyzer = MaterialsAnalyzer(
        jid="JVASP-1002",
        calculator_type="chgnet",
        properties_to_calculate=[
            "relax_structure",
            "calculate_formation_energy",
        ],
        chemical_potentials_file="../chipsff/chemical_potentials.json",
    )
    analyzer.job_info["equilibrium_energy"] = -100.0
    relaxed_atoms = analyzer.relax_structure()
    formation_energy = analyzer.calculate_formation_energy(relaxed_atoms)
    assert formation_energy is not None
    assert isinstance(formation_energy, float)