import re
import json
from utils import cleanup


class PromptGenerator:
    def __init__(self, model, tokenizer, translator, system_prompt, template, enrichment_model, enrichment_tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.translator = translator
        self.system_prompt = system_prompt
        self.template = template
        self.device = device
        self.DICT_REGEXP = re.compile('{.*?}')
        self.RESPONSE_REGEXP = re.compile('(?<=\[/INST]).*')
        self.PUNCT_TO_DEL_REGEXP = re.compile('}|{|\]|\[|"')
        self.ENR_END_TOKEN = '\n<|endoftext|>'
        #self.KEYWORDS = ['beautiful','high resolution', 'masterpiece', 'ultra-detailed', 'high quality', 'best quality']
        self.enrichment_model = enrichment_model
        self.enrichment_tokenizer = enrichment_tokenizer
        self.model.to(self.device)
        self.enrichment_model.to(self.device)
    
    def translate(self, text):
        sentences = text.split('.')
        translated_sentences = []
        for sentence in sentences:
            translated_sentence = self.translator.translate(sentence)
            translated_sentences.append(translated_sentence)
        return '. '.join(translated_sentences)
    
    def make_llm_prompt(self, text):
        return self.template.format(system_prompt = self.system_prompt, user_message=text)

    def generate_text(self, text):
        input_ids = self.tokenizer(text, return_tensors='pt').input_ids.to('cuda')
        output = self.tokenizer.decode(self.model.generate(input_ids, max_new_tokens = 50)[0])
        cleanup()
        return output
    
    def process_llm_output(self, llm_output):
        llm_output = re.sub('\n', '', llm_output)
        llm_output = llm_output.replace('</s>', '')
        llm_output += '"}'
        return llm_output
    
    def extract(self, llm_output, regexp):
        matches = regexp.findall(llm_output)
        if matches:
            return matches[0]
    
    def make_prompt_from_broken_json(self, llm_output):
        llm_response = self.extract(llm_output, self.RESPONSE_REGEXP)
        if llm_response:
            prompt = self.PUNCT_TO_DEL_REGEXP.sub('', llm_response)
            return prompt
       
    def make_prompt_from_dict(self, llm_output):
        prompt_dict_json = self.extract(llm_output, self.DICT_REGEXP)
        prompt_dict_json = prompt_dict_json.lower()
        prompt_dict = json.loads(prompt_dict_json)
        prompt = f'{prompt_dict.get("object", "")} {prompt_dict.get("action", "")}, '
        if type(prompt_dict.get('description', None)) == list:
            prompt += f'{", ".join(prompt_dict["description"])}, '
        elif type(prompt_dict.get('description', None)) == str:
            prompt += f'{prompt_dict["description"]}, '
        if 'background' in prompt_dict:
            prompt += f'{prompt_dict["background"]} background, '
        #prompt += f'{", ".join(self.KEYWORDS)}, '
        if 'lighting' in prompt_dict:
            prompt += f'{prompt_dict["lighting"]}, '
        if 'style' in prompt_dict:
            prompt += f'{prompt_dict["style"]}'
        
        prompt = self.PUNCT_TO_DEL_REGEXP.sub('', prompt)
        return prompt
    
    def enrich(self, prompt):
        input_ids = self.enrichment_tokenizer(prompt, return_tensors='pt').input_ids.to('cuda')
        output = self.enrichment_tokenizer.decode(self.enrichment_model.generate(input_ids, max_new_tokens = 30)[0])
        cleanup()
        output = output.replace(self.ENR_END_TOKEN, '')
        return output
    
    def generate(self, text):
        try:
            translated_text = self.translate(text)
        except:
            translated_text = text
        llm_prompt = self.make_llm_prompt(translated_text)
        llm_output = self.generate_text(llm_prompt)
        llm_output = self.process_llm_output(llm_output)
        try :
            prompt = self.make_prompt_from_dict(llm_output)
        except:
            prompt = self.make_prompt_from_broken_json(llm_output)
        if prompt:
            enriched_prompt = self.enrich(prompt)
            return enriched_prompt
        return text