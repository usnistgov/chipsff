from typing import List
from typing import Literal
from typing import Dict
from pydantic_settings import BaseSettings
from typing import Optional



class CHIPSFFConfig(BaseSettings):
    # Generator config
    jid: Optional[str] = None
    jid_list: Optional[List[str]] = None
    calculator_type: str = "alignn_ff"
    calculator_types: Optional[List[str]] = ["alignn_ff"]
    chemical_potentials_file: str = "chemical_potentials.json"
    film_id: str = ""
    film_index: str = "0_0_1"
    substrate_id: str = ""
    substrate_index: str = "0_0_1"

