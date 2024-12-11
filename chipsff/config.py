from typing import List, Optional, Dict
from pydantic_settings import BaseSettings
from pydantic import Field


class CHIPSFFConfig(BaseSettings):
    jid: Optional[str] = None
    jid_list: Optional[List[str]] = None
    calculator_type: Optional[str] = (
        None  # Changed to Optional to allow multiple calculators
    )
    calculator_types: Optional[List[str]] = (
        None  # Optional list for multiple calculators
    )
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
                "constant_volume": False,
            },
        },
        description="Settings for bulk relaxation including filter type and relaxation parameters",
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
                "steps": 200,
            },
        },
        description="Settings for surfaces including layer count, vacuum, and relaxation settings",
    )

    # Defect settings with defect generation and relaxation settings
    defect_settings: Dict = Field(
        default_factory=lambda: {
            "generate_settings": {
                "on_conventional_cell": True,
                "enforce_c_size": 8,
                "extend": 1,
            },
            "relaxation_settings": {
                "filter_type": "ExpCellFilter",
                "constant_volume": True,
                "fmax": 0.05,
                "steps": 200,
            },
        },
        description="Settings for defect generation and relaxation",
    )

    # Phonon and MD settings
    phonon_settings: Dict = Field(
        default_factory=lambda: {"dim": [2, 2, 2], "distance": 0.2},
        description="Phonon calculation settings",
    )

    phonon3_settings: Dict = Field(
        default_factory=lambda: {"dim": [2, 2, 2], "distance": 0.2},
        description="Third-order phonon calculation settings",
    )

    md_settings: Dict = Field(
        default_factory=lambda: {
            "dt": 1,
            "temp0": 3500,
            "nsteps0": 1000,
            "temp1": 300,
            "nsteps1": 2000,
            "taut": 20,
            "min_size": 10.0,
        },
        description="Settings for molecular dynamics (MD) simulation",
    )

    # Add mlearn_elements field
    mlearn_elements: Optional[List[str]] = Field(
        default_factory=lambda: [],
        description="List of elements to compare forces with mlearn dataset",
    )

    alignn_ff_db: bool = False
    mptrj: bool = False
    num_samples: Optional[int] = Field(
        default=None,
        description="Number of samples to process from the alignn_ff_db dataset or mptrj dataset",
    )

    # Calculator-specific settings
    calculator_settings: Dict[str, Dict] = Field(
        default_factory=dict,
        description="Calculator-specific settings. Keys are calculator types.",
    )

    properties_to_calculate: List[str] = Field(
        default_factory=lambda: [
            "relax_structure",
            "calculate_ev_curve",
            "calculate_formation_energy",
            "calculate_forces",
            "calculate_elastic_tensor",
            "run_phonon_analysis",
            "analyze_surfaces",
            "analyze_defects",
            "run_phonon3_analysis",
            "general_melter",
            "compare_mlearn_forces",
            "calculate_rdf",
        ],
        description="List of properties to calculate in the analysis",
    )

    scaling_test: Optional[bool] = Field(
        default=False, description="Whether to perform the scaling test"
    )
    scaling_numbers: Optional[List[int]] = Field(
        default_factory=lambda: [1, 2, 3, 4, 5],
        description="List of scaling factors for supercell sizes",
    )
    scaling_element: Optional[str] = Field(
        default="Cu", description="Element symbol to use for the scaling test"
    )
    scaling_calculators: Optional[List[str]] = Field(
        default_factory=lambda: [],
        description="List of calculator types to test in the scaling analysis",
    )
