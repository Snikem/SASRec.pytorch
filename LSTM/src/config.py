from dataclasses import dataclass


@dataclass
class QuantConfig:
    method: str = "fp32"
    bits: int = 8

    pact_alpha_init: float = 10.0

    apot_num_powers: int = 8
    apot_max_addends: int = 3
    apot_alpha_init: float = 1.0

    ada_val: bool = False

    tqt_alpha_init: float = 2.0
