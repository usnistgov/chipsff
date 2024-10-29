import pytest
from chipsff.run_chipsff import MaterialsAnalyzer
from chipsff.config import CHIPSFFConfig

def test_materials_analyzer_initialization():
    # Define the test path for chemical potentials
    config = CHIPSFFConfig(
        jid='JVASP-1002',
        calculator_type='chgnet',
        properties_to_calculate=['relax_structure'],
        chemical_potentials_file='../chipsff/chemical_potentials.json'  # Replace with actual path
    )
    analyzer = MaterialsAnalyzer(
        jid=config.jid,
        calculator_type=config.calculator_type,
        properties_to_calculate=config.properties_to_calculate,
        chemical_potentials_file=config.chemical_potentials_file
    )
    assert analyzer.jid == 'JVASP-1002'
    assert analyzer.calculator_type == 'chgnet'

def test_relax_structure():
    # Define the test path for chemical potentials
    config = CHIPSFFConfig(
        jid='JVASP-1002',
        calculator_type='chgnet',
        properties_to_calculate=['relax_structure'],
        chemical_potentials_file='chemical_potentials.json'  # Replace with actual path
    )
    analyzer = MaterialsAnalyzer(
        jid=config.jid,
        calculator_type=config.calculator_type,
        properties_to_calculate=config.properties_to_calculate,
        chemical_potentials_file=config.chemical_potentials_file
    )
    relaxed_atoms = analyzer.relax_structure()
    assert relaxed_atoms is not None
    assert hasattr(relaxed_atoms, 'lattice')
