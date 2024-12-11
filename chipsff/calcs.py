#!/usr/bin/env python


def setup_calculator(calculator_type, calculator_settings):
    """
    Initializes and returns the appropriate calculator based on the calculator type and its settings.

    Args:
        calculator_type (str): The type/name of the calculator.
        calculator_settings (dict): Settings specific to the calculator.

    Returns:
        calculator: An instance of the specified calculator.
    """
    if calculator_type == "matgl":
        import matgl
        from matgl.ext.ase import M3GNetCalculator

        model_name = calculator_settings.get("model", "M3GNet-MP-2021.2.8-PES")
        pot = matgl.load_model(model_name)
        compute_stress = calculator_settings.get("compute_stress", True)
        stress_weight = calculator_settings.get("stress_weight", 0.01)
        return M3GNetCalculator(
            pot, compute_stress=compute_stress, stress_weight=stress_weight
        )

    elif calculator_type == "matgl-direct":
        import matgl
        from matgl.ext.ase import M3GNetCalculator

        model_name = calculator_settings.get(
            "model", "M3GNet-MP-2021.2.8-DIRECT-PES"
        )
        pot = matgl.load_model(model_name)
        compute_stress = calculator_settings.get("compute_stress", True)
        stress_weight = calculator_settings.get("stress_weight", 0.01)
        return M3GNetCalculator(
            pot, compute_stress=compute_stress, stress_weight=stress_weight
        )

    elif calculator_type == "alignn_ff":
        from alignn.ff.ff import AlignnAtomwiseCalculator, default_path

        return AlignnAtomwiseCalculator()

    elif calculator_type == "chgnet":
        from chgnet.model.dynamics import CHGNetCalculator

        return CHGNetCalculator()

    elif calculator_type == "mace":
        from mace.calculators import mace_mp

        return mace_mp()

    elif calculator_type == "mace-alexandria":
        from mace.calculators.mace import MACECalculator

        # TODO: Make an option to provide path
        model_path = calculator_settings.get(
            "model_path",
            "/users/dtw2/utils/models/alexandria_v2/mace/2D_universal_force_field_cpu.model",
        )
        device = calculator_settings.get("device", "cpu")
        return MACECalculator(model_path, device=device)

    elif calculator_type == "sevennet":
        from sevenn.sevennet_calculator import SevenNetCalculator

        checkpoint_path = calculator_settings.get(
            "checkpoint_path",
            "/users/dtw2/SevenNet/pretrained_potentials/SevenNet_0__11July2024/checkpoint_sevennet_0.pth",
        )
        device = calculator_settings.get("device", "cpu")
        return SevenNetCalculator(checkpoint_path, device=device)

    elif calculator_type == "orb-v2":
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator

        orbff = pretrained.orb_v2()
        device = calculator_settings.get("device", "cpu")
        return ORBCalculator(orbff, device=device)

    elif calculator_type == "eqV2_31M_omat":
        from fairchem.core import OCPCalculator

        checkpoint_path = calculator_settings.get(
            "checkpoint_path",
            "/users/dtw2/fairchem-models/pretrained_models/eqV2_31M_omat.pt",
        )
        return OCPCalculator(checkpoint_path=checkpoint_path)

    elif calculator_type == "eqV2_86M_omat":
        from fairchem.core import OCPCalculator

        checkpoint_path = calculator_settings.get(
            "checkpoint_path",
            "/users/dtw2/fairchem-models/pretrained_models/eqV2_86M_omat.pt",
        )
        return OCPCalculator(checkpoint_path=checkpoint_path)

    elif calculator_type == "eqV2_153M_omat":
        from fairchem.core import OCPCalculator

        checkpoint_path = calculator_settings.get(
            "checkpoint_path",
            "/users/dtw2/fairchem-models/pretrained_models/eqV2_153M_omat.pt",
        )
        return OCPCalculator(checkpoint_path=checkpoint_path)

    elif calculator_type == "eqV2_31M_omat_mp_salex":
        from fairchem.core import OCPCalculator

        checkpoint_path = calculator_settings.get(
            "checkpoint_path",
            "/users/dtw2/fairchem-models/pretrained_models/eqV2_31M_omat_mp_salex.pt",
        )
        return OCPCalculator(checkpoint_path=checkpoint_path)

    elif calculator_type == "eqV2_86M_omat_mp_salex":
        from fairchem.core import OCPCalculator

        checkpoint_path = calculator_settings.get(
            "checkpoint_path",
            "/users/dtw2/fairchem-models/pretrained_models/eqV2_86M_omat_mp_salex.pt",
        )
        return OCPCalculator(checkpoint_path=checkpoint_path)

    else:
        raise ValueError(f"Unsupported calculator type: {calculator_type}")
