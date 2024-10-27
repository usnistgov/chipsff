from typing import List
from typing import Literal
from typing import Dict
from pydantic_settings import BaseSettings
from typing import Optional


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
    relaxation_settings: Optional[dict] = None
    phonon_settings: Optional[dict] = None
    properties_to_calculate: Optional[List[str]] = None
    surface_indices_list: Optional[List[List[int]]] = None
    use_conventional_cell: Optional[bool] = False
