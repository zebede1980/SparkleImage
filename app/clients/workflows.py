"""ComfyUI API-format workflow graphs.

Built from Comfy's official templates (Qwen-Image 2.1 edit) and the SeedVR2
custom node's own defaults. Node ids are arbitrary strings; every graph saves
through a SaveImage with id OUTPUT_NODE, which is where the result is read from.

Model filenames are the ones on the home PC. A missing file shows up as a
`/prompt` validation error naming it, which ComfyClient passes on verbatim.
"""

from __future__ import annotations

OUTPUT_NODE = "56"

QWEN_UNET = "qwen_image_2.1_int8_convrot.safetensors"
QWEN_CLIP = "qwen3vl_8b_int8_convrot.safetensors"
QWEN_VAE = "qwen_image_2.1_vae_bf16.safetensors"
SEEDVR2_DIT = "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors"
SEEDVR2_VAE = "ema_vae_fp16.safetensors"


def qwen_edit(source: str, instruction: str, seed: int, megapixels: float = 1.5, steps: int = 25) -> dict:
    """Instruction edit of one image. The output keeps the source's aspect ratio.

    The source is scaled to `megapixels` and snapped to the multiple of 32 the
    model needs; `resolution: 0` tells the encoder to take the reference at that
    size rather than rescaling it again. 1.5MP takes ~25s warm on a 16GB card;
    above ~2MP the model spills out of VRAM and takes minutes.
    """
    return {
        "37": {"class_type": "UNETLoader", "inputs": {"unet_name": QWEN_UNET, "weight_dtype": "default"}},
        "38": {"class_type": "CLIPLoader", "inputs": {"clip_name": QWEN_CLIP, "type": "qwen_image", "device": "default"}},
        "39": {"class_type": "VAELoader", "inputs": {"vae_name": QWEN_VAE}},
        "78": {"class_type": "LoadImage", "inputs": {"image": source}},
        "79": {
            "class_type": "ResizeImageMaskNode",
            "inputs": {"input": ["78", 0], "resize_type": "scale total pixels",
                       "resize_type.megapixels": megapixels, "scale_method": "area"},
        },
        "77": {
            "class_type": "ResizeImageMaskNode",
            "inputs": {"input": ["79", 0], "resize_type": "scale to multiple",
                       "resize_type.multiple": 32, "scale_method": "bicubic"},
        },
        "76": {"class_type": "GetImageSize", "inputs": {"image": ["77", 0]}},
        "62": {"class_type": "EmptyLatentImage", "inputs": {"width": ["76", 0], "height": ["76", 1], "batch_size": 1}},
        "64": {
            "class_type": "TextEncodeQwenImage21",
            "inputs": {"clip": ["38", 0], "vae": ["39", 0], "prompt": instruction, "negative_prompt": "",
                       "resolution": 0, "images.image_1": ["77", 0]},
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {"model": ["37", 0], "positive": ["64", 0], "negative": ["64", 1], "latent_image": ["62", 0],
                       "seed": seed, "steps": steps, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
                       "denoise": 1.0},
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["39", 0]}},
        OUTPUT_NODE: {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "sparkle/edit"}},
    }


def seedvr2_upscale(source: str, short_edge: int, seed: int = 42) -> dict:
    """SeedVR2 7B upscale to `short_edge` pixels on the shorter side.

    Block swapping and tiled VAE keep a ~3000x2200 output inside 16GB alongside
    whatever else ComfyUI has cached. `color_correction: lab` pins the output's
    colour to the input's, so the upscaler sharpens without regrading.
    """
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": source}},
        "2": {
            "class_type": "SeedVR2LoadDiTModel",
            "inputs": {"model": SEEDVR2_DIT, "device": "cuda:0", "offload_device": "cpu", "blocks_to_swap": 20},
        },
        "3": {
            "class_type": "SeedVR2LoadVAEModel",
            "inputs": {"model": SEEDVR2_VAE, "device": "cuda:0", "encode_tiled": True, "decode_tiled": True},
        },
        "4": {
            "class_type": "SeedVR2VideoUpscaler",
            "inputs": {"image": ["1", 0], "dit": ["2", 0], "vae": ["3", 0], "seed": seed,
                       "resolution": short_edge - short_edge % 2, "max_resolution": 0, "batch_size": 1,
                       "uniform_batch_size": False, "color_correction": "lab"},
        },
        OUTPUT_NODE: {"class_type": "SaveImage", "inputs": {"images": ["4", 0], "filename_prefix": "sparkle/upscale"}},
    }
