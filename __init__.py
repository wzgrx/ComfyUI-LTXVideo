from .nodes_registry import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .prompt_enhancer_nodes import (  # noqa: F401
    LTXVPromptEnhancer,
    LTXVPromptEnhancerLoader,
)
from .stg import LTXVApplySTG, STGGuiderNode  # noqa: F401
from .tricks import NODE_CLASS_MAPPINGS as TRICKS_NODE_CLASS_MAPPINGS
from .tricks import NODE_DISPLAY_NAME_MAPPINGS as TRICKS_NODE_DISPLAY_NAME_MAPPINGS

# Merge the node mappings from tricks into the main mappings
NODE_CLASS_MAPPINGS.update(TRICKS_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(TRICKS_NODE_DISPLAY_NAME_MAPPINGS)

# Export so that ComfyUI can pick them up.
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
