import os

MODEL_GEN={
    "prompt_generators": [
        {
            "model": "Gustavosta/MagicPrompt-Stable-Diffusion",
            "tokenizer": "gpt2",
            "device": -1
        }
    ],
    "diffusers": [
        {
            "path": "stabilityai/stable-diffusion-xl-base-1.0",
            "use_safetensors": True,
            "variant": "fp16",
            "pipeline": "StableDiffusionXLPipeline"
        },
        {
            "path": "SG161222/RealVisXL_V4.0",
            "use_safetensors": True,
            "variant": "fp16",
            "pipeline": "StableDiffusionXLPipeline"
        },
        {
            "path": "Corcelio/mobius",
            "use_safetensors": True,
            "pipeline": "StableDiffusionXLPipeline"
        }
    ]
}


DATASET_META = {
    "real": [
        {"path": "abcd10987/open-images-v7", "create_splits": False},
        {"path": "abcd10987/ffhq-256", "create_splits": False},
        {"path": "abcd10987/celeb-a-hq", "create_splits": False}
    ],
    "fake": [
        {"path": "abcd10987/bm-realvisxl", "create_splits": False},
        {"path": "abcd10987/bm-mobius", "create_splits": False},
        {"path": "abcd10987/bm-sdxl", "create_splits": False},
        {"path": "abcd10987/syn_sdxl", "create_splits": False},
        {"path": "abcd10987/syn_realvis", "create_splits": False},
        {"path": "abcd10987/syn_mobius", "create_splits": False},
        {"path": "abcd10987/celeb-a-hq___stable-diffusion-xl-base-1.0", "create_splits": False},
        {"path": "abcd10987/ffhq-256___stable-diffusion-xl-base-1.0","create_splits": False},
    ]
}

FACE_TRAINING_DATASET_META = {
    "real": [
        {"path": "abcd10987/celeb-a-hq_training_faces", "create_splits": False},
        {"path": "abcd10987/ffhq-256_training_faces", "create_splits": False},
    ],
    "fake": [
        {"path": "abcd10987/celeb-a-hq___stable-diffusion-xl-base-1.0___256_training_faces", "create_splits": False},
        {"path": "abcd10987/ffhq-256___stable-diffusion-xl-base-1.0_training_faces", "create_splits": False}
    ]
}


HUGGINGFACE_CACHE_DIR = os.path.expanduser('~/.cache/huggingface')

TARGET_IMAGE_SIZE = (256, 256)

PROMPT_TYPES = ('random', 'annotation')

PROMPT_GENERATOR_ARGS = {
    m['model']: m for m in MODEL_GEN['prompt_generators']
}

PROMPT_GENERATOR_NAMES = list(PROMPT_GENERATOR_ARGS.keys())

DIFFUSER_ARGS = {
    m['path']: {k: v for k, v in m.items() if k != 'path' and k != 'pipeline'}  
    for m in MODEL_GEN['diffusers']
}

DIFFUSER_PIPELINE = {
    m['path']: m['pipeline'] for m in MODEL_GEN['diffusers'] if 'pipeline' in m
}

DIFFUSER_NAMES = list(DIFFUSER_ARGS.keys())

IMAGE_ANNOTATION_MODEL = "Salesforce/blip2-opt-2.7b-coco"
