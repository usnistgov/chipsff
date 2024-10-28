from typing import List
from typing import Literal
from typing import Dict
from pydantic_settings import BaseSettings
from typing import Optional
from pydantic import Field


class CHIPSFFConfig(BaseSettings):
    jid: Optional[str] = None
    jid_list: Optional[List[str]] = None
    calculator_type: str = "alignn_ff"
    calculator_types: Optional[List[str]] = ["alignn_ff"]
    chemical_potentials_file: str = "chemical_potentials.json"
    film_id: Optional[List[str]] = None
    film_index: str = "0_0_1"
    substrate_id: Optional[List[str]] = None
    substrate_index: str = "0_0_1"
    use_conventional_cell: Optional[bool] = False
    
    # Bulk relaxation settings
    bulk_relaxation_settings: Dict = Field(
        default_factory=lambda: {
            "filter_type": "ExpCellFilter",
            "relaxation_settings": {
                "fmax": 0.05,
                "steps": 200,
                "constant_volume": False
            }
        },
        description="Settings for bulk relaxation including filter type and relaxation parameters"
    )

    # Surface settings including indices list
    surface_settings: Dict = Field(
        default_factory=lambda: {
            "indices_list": [
                [1, 0, 0],
                [1, 1, 1],
                [1, 1, 0],
                [0, 1, 1],
                [0, 0, 1],
                [0, 1, 0],
            ],
            "layers": 4,
            "vacuum": 18,
            "relaxation_settings": {
                "filter_type": "ExpCellFilter",
                "constant_volume": True,
                "fmax": 0.05,
                "steps": 200
            }
        },
        description="Settings for surfaces including layer count, vacuum, and relaxation settings"
    )

    # Defect settings with defect generation and relaxation settings
    defect_settings: Dict = Field(
        default_factory=lambda: {
            "generate_settings": {
                "on_conventional_cell": True,
                "enforce_c_size": 8,
                "extend": 1
            },
            "relaxation_settings": {
                "filter_type": "ExpCellFilter",
                "constant_volume": True,
                "fmax": 0.05,
                "steps": 200
            }
        },
        description="Settings for defect generation and relaxation"
    )

    # Phonon and MD settings
    phonon_settings: Dict = Field(
        default_factory=lambda: {
            "dim": [2, 2, 2],
            "distance": 0.2
        },
        description="Phonon calculation settings"
    )
    
    phonon3_settings: Dict = Field(
        default_factory=lambda: {
            "dim": [2, 2, 2],
            "distance": 0.2
        },
        description="Third-order phonon calculation settings"
    )

    md_settings: Dict = Field(
        default_factory=lambda: {
            "dt": 1,
            "temp0": 3500,
            "nsteps0": 1000,
            "temp1": 300,
            "nsteps1": 2000,
            "taut": 20,
            "min_size": 10.0
        },
        description="Settings for molecular dynamics (MD) simulation"
    )

    properties_to_calculate: List[str] = Field(
        default_factory=lambda: [
            "relax_structure",
            "calculate_ev_curve",
            "calculate_formation_energy",
            "calculate_elastic_tensor",
            "run_phonon_analysis",
            "analyze_surfaces",
            "analyze_defects",
            "run_phonon3_analysis",
            "general_melter",
            "calculate_rdf"
        ],
        description="List of properties to calculate in the analysis"
    )
