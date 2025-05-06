from .decoder_noise import DecoderNoise  # noqa: F401
from .easy_samplers import LTXVBaseSampler  # noqa: F401
from .film_grain import LTXVFilmGrain  # noqa: F401
from .guide import LTXVAddGuideAdvanced  # noqa: F401
from .latent_adain import LTXVAdainLatent  # noqa: F401
from .latent_upsampler import LTXVLatentUpsampler  # noqa: F401
from .latents import LTXVSelectLatents, LTXVSetVideoLatentNoiseMasks  # noqa: F401
from .nodes_registry import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .prompt_enhancer_nodes import (  # noqa: F401
    LTXVPromptEnhancer,
    LTXVPromptEnhancerLoader,
)
from .q8_nodes import LTXVQ8Patch  # noqa: F401
from .recurrent_sampler import (  # noqa: F401
    LinearOverlapLatentTransition,
    LTXVRecurrentKSampler,
)
from .stg import (  # noqa: F401
    LTXVApplySTG,
    STGAdvancedPresetsNode,
    STGGuiderAdvancedNode,
    STGGuiderNode,
)
from .tiled_sampler import LTXVTiledSampler  # noqa: F401
from .tricks import NODE_CLASS_MAPPINGS as TRICKS_NODE_CLASS_MAPPINGS
from .tricks import NODE_DISPLAY_NAME_MAPPINGS as TRICKS_NODE_DISPLAY_NAME_MAPPINGS

# Merge the node mappings from tricks into the main mappings
NODE_CLASS_MAPPINGS.update(TRICKS_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(TRICKS_NODE_DISPLAY_NAME_MAPPINGS)

# Export so that ComfyUI can pick them up.
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
