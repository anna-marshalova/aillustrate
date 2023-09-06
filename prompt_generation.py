import os
import json
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

class PromptGenerator:
    def __init__(self):
        with open('config.json') as config_file:
            self.config = json.load(config_file)
        with open('llm_prompt_template.txt') as f:
            self.llm_prompt_template = f.read()
        self.token = os.environ['HF_TOKEN']
        self.model_name = self.config['LANGUAGE_MODEL']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16,
                                                     low_cpu_mem_usage=True, token=self.token).to(self.device)

    def make_prompt(self, text, template:str = None):
        if not template:
            template = self.llm_prompt_template
        return template.format(text=text)

    def generate(self, text):
        prompt = self.make_prompt(text)
        input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.to(self.device)
        output = self.tokenizer.decode(self.model.generate(input_ids)[0])
        return output
