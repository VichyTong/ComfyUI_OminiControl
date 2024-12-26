import torch
from PIL import Image
import numpy as np
from diffusers import FluxPipeline, FluxTransformer2DModel
from ComfyUI_RH_OminiControl.src.generate import generate, seed_everything
from ComfyUI_RH_OminiControl.src.condition import Condition
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel,T5TokenizerFast
import folder_paths
import os
from ComfyUI_RH_OminiControl.rh_utils import *

def run(t_img, prompt, seed):

    assert t_img.shape[0] == 1
    
    i = 255. * t_img[0].numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8)).convert("RGB")
    
    # Pad to 1:1 aspect ratio
    w, h = img.size
    if w > h:
        new_h = w
        padding = (w - h) // 2
        padded_img = Image.new('RGB', (w, new_h), (0, 0, 0))
        padded_img.paste(img, (0, padding))
    else:
        new_w = h 
        padding = (h - w) // 2
        padded_img = Image.new('RGB', (new_w, h), (0, 0, 0))
        padded_img.paste(img, (padding, 0))
        
    # Resize to target dimensions
    image = padded_img.resize((g_width, g_height))

    release_gpu()

    flux_dir = os.path.join(folder_paths.models_dir, 'flux', 'FLUX.1-schnell')
    lora_model = os.path.join(folder_paths.models_dir, 'flux', 'OminiControl', 'omini', 'subject_512.safetensors')

    encoded_condition = encode_condition(flux_dir, image)

    text_encoder = CLIPTextModel.from_pretrained(
        flux_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    text_encoder_2 = T5EncoderModel.from_pretrained(
        flux_dir, subfolder="text_encoder_2", torch_dtype=torch.bfloat16
    )
    tokenizer = CLIPTokenizer.from_pretrained(flux_dir, subfolder="tokenizer")
    tokenizer_2 = T5TokenizerFast.from_pretrained(flux_dir, subfolder="tokenizer_2")

    pipeline = FluxPipeline.from_pretrained(
        flux_dir,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        transformer=None,
        vae=None,
    ).to("cuda")

    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
            prompt=prompt, prompt_2=None, max_sequence_length=256
        )

    del text_encoder
    del text_encoder_2
    del tokenizer
    del tokenizer_2
    del pipeline

    release_gpu()

    pipeline = FluxPipeline.from_pretrained(
        flux_dir,
        # transformer=transformer,
        text_encoder=None,
        text_encoder_2=None,
        tokenizer=None,
        tokenizer_2=None,
        vae=None,
        torch_dtype=torch.bfloat16,
    )

    pipeline.to('cuda')

    pipeline.load_lora_weights(
        lora_model,
        adapter_name="subject",
    )

    condition = Condition("subject", image)

    seed_everything()

    result_latents = generate(
    # result_img = generate(
        pipeline,
        encoded_condition = encoded_condition,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        text_ids=text_ids,
        conditions=[condition],
        output_type="latent",
        return_dict=False,
        num_inference_steps=8,
        height=g_height,
        width=g_width,
        generator=torch.Generator(device='cuda').manual_seed(seed),
    )

    del pipeline

    release_gpu()

    result_img = decode_latents(flux_dir, result_latents[0]).images[0]

    return torch.from_numpy(np.array(result_img).astype(np.float32) / 255.0).unsqueeze(0)



    