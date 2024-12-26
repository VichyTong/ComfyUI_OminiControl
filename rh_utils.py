import torch
from diffusers import FluxPipeline
import gc
from ComfyUI_RH_OminiControl.src.generate import generate, seed_everything
from ComfyUI_RH_OminiControl.src.condition import Condition

from diffusers.pipelines.flux.pipeline_flux import (
    FluxPipelineOutput,
)

g_width = 512
g_height = 512

def release_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

def encode_condition(flux_dir, image, condition_type='subject'):
    pipeline = FluxPipeline.from_pretrained(
        flux_dir,
        text_encoder=None,
        text_encoder_2=None,
        tokenizer=None,
        tokenizer_2=None,
        transformer=None,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    condition = Condition(condition_type, image)
    tokens, ids, type_id = condition.encode(pipeline)

    del condition
    del pipeline
    release_gpu()

    return (tokens, ids, type_id)

def decode_latents(flux_dir, latents):
    pipeline = FluxPipeline.from_pretrained(
        flux_dir,
        text_encoder=None,
        text_encoder_2=None,
        tokenizer=None,
        tokenizer_2=None,
        transformer=None,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    latents = pipeline._unpack_latents(latents, g_width, g_height, pipeline.vae_scale_factor)
    latents = (
        latents / pipeline.vae.config.scaling_factor
    ) + pipeline.vae.config.shift_factor
    image = pipeline.vae.decode(latents, return_dict=False)[0]
    image = pipeline.image_processor.postprocess(image, output_type="pil")

    return FluxPipelineOutput(images=image)