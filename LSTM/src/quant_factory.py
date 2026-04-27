from .quantizers import (
    IdentityQuantizer,
    LSQQuantizer,
    PACTQuantizer,
    TQTQuantizer,
    APoTQuantizer,
    ADAQuantizer,
)

class FabricQuantizer:
    def __init__(self, config):
        self.config = config

    def make_activations(self, name: str = ""):
        method = self.config.method
        
        if method == "fp32":
            return IdentityQuantizer()
        if method == "lsq":
            return LSQQuantizer(
                bits=self.config.bits,
            )
        if method == "pact":
            return PACTQuantizer(
                bits=self.config.bits,
                alpha_init=self.config.pact_alpha_init,
            )
        if method == "apot":
            return APoTQuantizer(
                bits=self.config.bits,
                max_addends=self.config.apot_max_addends,
                num_powers=self.config.apot_num_powers,
                alpha_init=self.config.apot_alpha_init
            )
        if method == "ada":
            return IdentityQuantizer()

        if method == "tqt":
            return TQTQuantizer(
                bits=self.config.bits,
                alpha_init=self.config.tqt_alpha_init
            )

    def make_weights(self, weight, name: str = ""):
        method = self.config.method
        
        if method == "fp32":
            return IdentityQuantizer()
        if method == "lsq":
            return LSQQuantizer(
                bits=self.config.bits,
            )
        if method == "pact":
            return IdentityQuantizer() # Потому что на активациях
            
        if method == "apot":
            return APoTQuantizer(
                bits=self.config.bits,
                max_addends=self.config.apot_max_addends,
                num_powers=self.config.apot_num_powers,
                alpha_init=self.config.apot_alpha_init
            )
        if method == "ada":
            return ADAQuantizer(
                weight,
                bits=self.config.bits,
                val=self.config.ada_val
            )
        if method == "tqt":
            alpha_init = weight.detach().abs().max()
            alpha_init = max(alpha_init, 1e-8)
            return TQTQuantizer(
                bits=self.config.bits,
                alpha_init=alpha_init
            )
