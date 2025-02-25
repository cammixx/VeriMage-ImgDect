"""AI model configurations"""

DEFAULT_CONFIG = {
    'model_paths': {
        'blip': "Salesforce/blip-image-captioning-base",
        'stable_diffusion': "runwayml/stable-diffusion-v1-5",
        'clip': "openai/clip-vit-base-patch32"
    },
    'thresholds': {
        'similarity': 0.8,
        'ela': 25
    }
} 