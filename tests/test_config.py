from chipsff.config import CHIPSFFConfig

def test_default_config():
    config = CHIPSFFConfig()
    assert config.calculator_type == 'alignn_ff'
    assert isinstance(config.properties_to_calculate, list)

