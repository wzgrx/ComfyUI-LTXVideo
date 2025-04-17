import torch
from comfy_extras.nodes_custom_sampler import CFGGuider, SamplerCustomAdvanced
from comfy_extras.nodes_lt import LTXVAddGuide, LTXVCropGuides

from .latents import LTXVAddLatents, LTXVSelectLatents
from .nodes_registry import comfy_node
from .tricks import AddLatentGuideNode


@comfy_node(description="Linear transition with overlap")
class LinearOverlapLatentTransition:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples1": ("LATENT",),
                "samples2": ("LATENT",),
                "overlap": ("INT", {"default": 1, "min": 1, "max": 256}),
            },
            "optional": {
                "axis": ("INT", {"default": 0}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "process"

    CATEGORY = "Lightricks/latent"

    def get_subbatch(self, samples):
        s = samples.copy()
        samples = s["samples"]
        return samples

    def process(self, samples1, samples2, overlap, axis=0):
        samples1 = self.get_subbatch(samples1)
        samples2 = self.get_subbatch(samples2)

        # Create transition coefficients
        alpha = torch.linspace(1, 0, overlap + 2)[1:-1].to(samples1.device)

        # Create shape for broadcasting based on the axis
        shape = [1] * samples1.dim()
        shape[axis] = alpha.size(0)
        alpha = alpha.reshape(shape)

        # Create slices for the overlap regions
        slice_all = [slice(None)] * samples1.dim()
        slice_overlap1 = slice_all.copy()
        slice_overlap1[axis] = slice(-overlap, None)
        slice_overlap2 = slice_all.copy()
        slice_overlap2[axis] = slice(0, overlap)
        slice_rest1 = slice_all.copy()
        slice_rest1[axis] = slice(None, -overlap)
        slice_rest2 = slice_all.copy()
        slice_rest2[axis] = slice(overlap, None)

        # Combine samples
        parts = [
            samples1[tuple(slice_rest1)],
            alpha * samples1[tuple(slice_overlap1)]
            + (1 - alpha) * samples2[tuple(slice_overlap2)],
            samples2[tuple(slice_rest2)],
        ]

        combined_samples = torch.cat(parts, dim=axis)
        combined_batch_index = torch.arange(0, combined_samples.shape[0])

        return (
            {
                "samples": combined_samples,
                "batch_index": combined_batch_index,
            },
        )


@comfy_node(
    name="LTXVRecurrentKSampler",
)
class LTXVRecurrentKSampler:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "noise": ("NOISE",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "latents": ("LATENT",),
                "chunk_sizes": ("STRING", {"default": "3", "multiline": False}),
                "overlaps": ("STRING", {"default": "1", "multiline": False}),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "input_image": ("IMAGE",),
                "linear_blend_latents": ("BOOLEAN", {"default": True}),
                "conditioning_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
            },
            "optional": {
                "guider": ("GUIDER",),
            },
        }

    RETURN_TYPES = (
        "LATENT",
        "LATENT",
    )
    RETURN_NAMES = (
        "output",
        "denoised_output",
    )

    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(
        self,
        model,
        vae,
        noise,
        sampler,
        sigmas,
        latents,
        chunk_sizes,
        overlaps,
        positive,
        negative,
        input_image,
        linear_blend_latents,
        conditioning_strength,
        guider=None,
    ):
        select_latents = LTXVSelectLatents().select_latents
        add_latent_guide = AddLatentGuideNode().generate
        add_latents = LTXVAddLatents().add_latents
        positive_orig = positive.copy()
        negative_orig = negative.copy()

        # Parse chunk sizes and overlaps from strings
        chunk_sizes = [int(x) for x in chunk_sizes.split(",")]
        overlaps = [int(x) for x in overlaps.split(",")]

        # Extend lists if shorter than number of sigma steps
        n_steps = len(sigmas) - 1
        if len(chunk_sizes) < n_steps:
            chunk_sizes.extend([chunk_sizes[-1]] * (n_steps - len(chunk_sizes)))
        if len(overlaps) < n_steps:
            overlaps.extend([overlaps[-1]] * (n_steps - len(overlaps)))

        # Initialize working latents
        current_latents = latents.copy()
        t_latents = None
        # Loop through sigma pairs for progressive denoising
        for i in range(n_steps):
            current_sigmas = sigmas[i : i + 2]
            current_chunk_size = chunk_sizes[i]
            current_overlap = overlaps[i]

            print(f"\nProcessing sigma step {i} with sigmas {current_sigmas}")
            print(
                f"Using chunk size {current_chunk_size} and overlap {current_overlap}"
            )

            # Calculate valid chunk starts to ensure the last chunk isn't shorter than the overlap
            total_frames = current_latents["samples"].shape[2]
            chunk_stride = current_chunk_size - current_overlap
            valid_chunk_starts = list(
                range(0, total_frames - current_overlap, chunk_stride)
            )

            # If the last chunk would be too short, remove the last start position
            if (
                total_frames > chunk_stride
                and (total_frames - valid_chunk_starts[-1]) < current_chunk_size
            ):
                print(
                    "last chunk is too short, it will only be of size",
                    total_frames - valid_chunk_starts[-1],
                    "frames",
                )

            # Process each chunk for current sigma pair
            for i_chunk, chunk_start in enumerate(valid_chunk_starts):
                (latents_chunk,) = select_latents(
                    current_latents, chunk_start, chunk_start + current_chunk_size - 1
                )
                print(f"Processing chunk {i_chunk} starting at frame {chunk_start}")

                if i_chunk == 0:
                    positive, negative, latents_chunk = LTXVAddGuide().generate(
                        positive_orig,
                        negative_orig,
                        vae,
                        latents_chunk,
                        input_image,
                        0,
                        0.75,
                    )
                else:
                    (cond_latent,) = select_latents(t_latents, -current_overlap, -1)
                    model, latents_chunk = add_latent_guide(
                        model, latents_chunk, cond_latent, 0, conditioning_strength
                    )

                if guider is None:
                    (guider_obj,) = CFGGuider().get_guider(
                        model, positive, negative, 1.0
                    )
                else:
                    guider_obj = guider
                (_, denoised_latents) = SamplerCustomAdvanced().sample(
                    noise, guider_obj, sampler, current_sigmas, latents_chunk
                )
                (positive, negative, denoised_latents) = LTXVCropGuides().crop(
                    positive, negative, denoised_latents
                )

                if i_chunk == 0:
                    t_latents = denoised_latents
                else:
                    if linear_blend_latents and current_overlap > 1:
                        # the first output latent is the result of a 1:8 latent
                        # reinterpreted as a 1:1 latent, so we ignore it
                        (denoised_latents_drop_first,) = select_latents(
                            denoised_latents, 1, -1
                        )
                        (t_latents,) = LinearOverlapLatentTransition().process(
                            t_latents,
                            denoised_latents_drop_first,
                            current_overlap - 1,
                            axis=2,
                        )
                    else:
                        (truncated_denoised_latents,) = select_latents(
                            denoised_latents, current_overlap, -1
                        )
                        (t_latents,) = add_latents(
                            t_latents, truncated_denoised_latents
                        )

                print(
                    f"Completed chunk {i_chunk}, current output shape: {t_latents['samples'].shape}"
                )

            # Update current_latents for next sigma step
            current_latents = t_latents.copy()
            print(f"Completed sigma step {i}")

        return t_latents, t_latents
