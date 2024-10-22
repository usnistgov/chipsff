from typing import List
from typing import Literal
from typing import Dict
from pydantic_settings import BaseSettings



class CHIPSFFConfig(BaseSettings):
    # Generator config
    jid: str = "JVASP-1002"
    jid_list: str = "JVASP-1002 JVASP-816"
    calculator_type: str = "alignn_ff"
    calculator_types: str = "matgl alignn_ff"
    chemical_potential_file: str = "chemical_potentials.json"
    film_id: str = "JVASP-1002"
    film_index: str = "0_0_1"
    substrate_id: str = "JVASP-1002"
    substrate_index: str = "0_0_1"

