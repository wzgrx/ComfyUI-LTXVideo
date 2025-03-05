import contextlib
from dataclasses import dataclass
from typing import List

import comfy.ldm.modules.attention
import comfy.samplers
import comfy.utils
import torch
from comfy.model_patcher import ModelPatcher

from .nodes_registry import comfy_node


def stg(
    noise_pred_pos,
    noise_pred_neg,
    noise_pred_pertubed,
    cfg_scale,
    stg_scale,
    rescale_scale,
):
    noise_pred = (
        noise_pred_neg
        + cfg_scale * (noise_pred_pos - noise_pred_neg)
        + stg_scale * (noise_pred_pos - noise_pred_pertubed)
    )
    if rescale_scale != 0:
        factor = noise_pred_pos.std() / noise_pred.std()
        factor = rescale_scale * factor + (1 - rescale_scale)
        noise_pred = noise_pred * factor
    return noise_pred


@comfy_node(name="LTXVApplySTG")
class LTXVApplySTG:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {"tooltip": "The model to apply the STG to."},
                ),
                "block_indices": (
                    "STRING",
                    {
                        "default": "14, 19",
                        "tooltip": "Comma-separated indices of the blocks to apply the STG to.",
                    },
                ),
            }
        }

    FUNCTION = "apply_stg"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    CATEGORY = "lightricks/LTXV"

    def apply_stg(self, model: ModelPatcher, block_indices: str):
        skip_block_list = [int(i.strip()) for i in block_indices.split(",")]
        new_model = model.clone()

        if "skip_block_list" in new_model.model_options["transformer_options"]:
            skip_block_list.extend(
                new_model.model_options["transformer_options"]["skip_block_list"]
            )
        new_model.model_options["transformer_options"][
            "skip_block_list"
        ] = skip_block_list

        return (new_model,)


@dataclass
class STGFlag:
    do_skip: bool = False
    skip_layers: List[int] = None


# context manager that replaces the attention function in a transformer block
class PatchAttention(contextlib.AbstractContextManager):
    def __init__(self, attn_idx=0):
        self.original_attention = comfy.ldm.modules.attention.optimized_attention
        self.original_attention_masked = (
            comfy.ldm.modules.attention.optimized_attention_masked
        )
        self.current_idx = -1
        self.attn_idx = attn_idx

    def __enter__(self):
        comfy.ldm.modules.attention.optimized_attention = self.stg_attention
        comfy.ldm.modules.attention.optimized_attention_masked = (
            self.stg_attention_masked
        )

    def __exit__(self, exc_type, exc_value, traceback):
        comfy.ldm.modules.attention.optimized_attention = self.original_attention
        comfy.ldm.modules.attention.optimized_attention_masked = (
            self.original_attention_masked
        )

    def stg_attention(self, q, k, v, heads, *args, **kwargs):
        self.current_idx += 1
        if self.current_idx == self.attn_idx:
            return v
        else:
            return self.original_attention(q, k, v, heads, *args, **kwargs)

    def stg_attention_masked(self, q, k, v, heads, *args, **kwargs):
        self.current_idx += 1
        if self.current_idx == self.attn_idx:
            return v
        else:
            return self.original_attention_masked(q, k, v, heads, *args, **kwargs)


class STGBlockWrapper:
    """Wraps transformer blocks to be able to skip attention layers."""

    def __init__(self, block, stg_flag: STGFlag, idx: int):
        self.flag = stg_flag
        self.idx = idx
        self.block = block

    def __call__(self, args, extra_args):
        context_manager = contextlib.nullcontext()
        if self.flag.do_skip and self.idx in self.flag.skip_layers:
            context_manager = PatchAttention(0)

        with context_manager:
            hidden_state = extra_args["original_block"](args)
        return hidden_state


class STGGuider(comfy.samplers.CFGGuider):
    def __init__(
        self, model: ModelPatcher, cfg, stg_scale, rescale_scale: float = None
    ):
        model = model.clone()
        super().__init__(model)

        self.stg_flag = STGFlag(
            do_skip=False,
            skip_layers=model.model_options["transformer_options"]["skip_block_list"],
        )

        self.patch_model(model, self.stg_flag)

        self.cfg = cfg
        self.stg_scale = stg_scale
        self.rescale_scale = rescale_scale

    @classmethod
    def patch_model(cls, model: ModelPatcher, stg_flag: STGFlag):
        transformer_blocks = cls.get_transformer_blocks(model)

        for i, block in enumerate(transformer_blocks):
            model.set_model_patch_replace(
                STGBlockWrapper(block, stg_flag, i), "dit", "double_block", i
            )

    @staticmethod
    def get_transformer_blocks(model: ModelPatcher):
        diffusion_model = model.get_model_object("diffusion_model")
        key = "diffusion_model.transformer_blocks"
        if diffusion_model.__class__.__name__ == "LTXVTransformer3D":
            key = "diffusion_model.transformer.transformer_blocks"
        return model.get_model_object(key)

    def set_conds(self, positive, negative):
        self.inner_set_conds({"positive": positive, "negative": negative})

    def predict_noise(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        model_options: dict = {},
        seed=None,
    ):
        # in CFGGuider.predict_noise, we call sampling_function(), which uses cfg_function() to compute pos & neg
        # but we'd rather do a single batch of sampling pos, neg, and perturbed, so we call calc_cond_batch([perturbed,pos,neg]) directly

        positive_cond = self.conds.get("positive", None)
        negative_cond = self.conds.get("negative", None)

        noise_pred_pos = comfy.samplers.calc_cond_batch(
            self.inner_model,
            [positive_cond],
            x,
            timestep,
            model_options,
        )[0]

        noise_pred_neg = 0
        noise_pred_perturbed = 0

        if self.cfg > 1:
            noise_pred_neg = comfy.samplers.calc_cond_batch(
                self.inner_model,
                [negative_cond],
                x,
                timestep,
                model_options,
            )[0]

        if self.stg_scale > 0:
            try:
                model_options["transformer_options"]["ptb_index"] = 0
                self.stg_flag.do_skip = True
                noise_pred_perturbed = comfy.samplers.calc_cond_batch(
                    self.inner_model,
                    [positive_cond],
                    x,
                    timestep,
                    model_options,
                )[0]
            finally:
                self.stg_flag.do_skip = False
                del model_options["transformer_options"]["ptb_index"]

        stg_result = stg(
            noise_pred_pos,
            noise_pred_neg,
            noise_pred_perturbed,
            self.cfg,
            self.stg_scale,
            self.rescale_scale,
        )

        # normally this would be done in cfg_function, but we skipped
        # that for efficiency: we can compute the noise predictions in
        # a single call to calc_cond_batch() (rather than two)
        # so we replicate the hook here
        for fn in model_options.get("sampler_post_cfg_function", []):
            args = {
                "denoised": stg_result,
                "cond": positive_cond,
                "uncond": negative_cond,
                "model": self.inner_model,
                "uncond_denoised": noise_pred_neg,
                "cond_denoised": noise_pred_pos,
                "sigma": timestep,
                "model_options": model_options,
                "input": x,
                # not in the original call in samplers.py:cfg_function, but made available for future hooks
                "perturbed_cond": positive_cond,
                "perturbed_cond_denoised": noise_pred_perturbed,
            }
            stg_result = fn(args)

        return stg_result


@comfy_node(name="STGGuider")
class STGGuiderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "stg": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01},
                ),
                "rescale": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("GUIDER",)

    FUNCTION = "get_guider"
    CATEGORY = "lightricks/LTXV"

    def get_guider(self, model, positive, negative, cfg, stg, rescale):
        guider = STGGuider(model, cfg, stg, rescale)
        guider.set_conds(positive, negative)
        return (guider,)
