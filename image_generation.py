import json
import torch
from diffusers import StableDiffusionPipeline

with open('config.json') as config_file:
    config = json.load(config_file)

class ImageGenerator:
    def __init__(self):
        with open('config.json') as config_file:
            self.config = json.load(config_file)
        self.model_name = self.config['IMAGE_GENERATION_MODEL']
        self.pipe = StableDiffusionPipeline.from_pretrained(self.model_name, torch_dtype=torch.float16)

    def generate_image(self, prompt):
        image = self.pipe(prompt).images[0]
        return image