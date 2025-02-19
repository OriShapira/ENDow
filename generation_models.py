from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging
from openai import AzureOpenAI

import config


HUGGINGFACE_ACCESS_TOKEN = config.HUGGINGFACE_ACCESS_TOKEN
AZURE_OPENAI_API_KEY = config.AZURE_OPENAI_API_KEY
AZURE_OPENAI_API_VERSION = config.AZURE_OPENAI_API_VERSION
AZURE_OPENAI_API_ENDPOINT = config.AZURE_OPENAI_API_ENDPOINT


class GenModelBase:

    def __init__(self):
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps:0' if torch.backends.mps.is_available() else 'cpu')
        self.HF_ACCESS_TOKEN = HUGGINGFACE_ACCESS_TOKEN
        self.load_model()

    def __str__(self):
        return f'GenModelBase - {self.model_name}'

    def max_context_len(self):
        return -1

    def load_model(self):
        logging.info(f'Loading model {self.model_name}')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, 
                                                       token=self.HF_ACCESS_TOKEN)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, 
                                                          token=self.HF_ACCESS_TOKEN, 
                                                          torch_dtype=torch.float16, 
                                                          device_map="auto")

    def generate(self, input_str):
        messages = [{"role": "user", "content": input_str}]
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, 
                                                       add_generation_prompt=True, 
                                                       return_tensors="pt")
        input_ids = input_ids.to('cuda')

        gen_tokens = self.model.generate(input_ids, max_new_tokens=500, 
                                         do_sample=True, temperature=0.3)
        gen_text = self.tokenizer.decode(gen_tokens[0])
        output = self.parse_output(gen_text)
        return output

    def parse_output(self, gen_text):
        return gen_text
    

class GenModelMistral7BInstruct(GenModelBase):
    def __init__(self):
        self.model_name = "mistralai/Mistral-7B-Instruct-v0.3"
        super().__init__()

    def __str__(self):
        return f'GenModelMistral7BInstruct - {self.model_name}'

    def max_context_len(self):
        return 6000

    def parse_output(self, gen_text):
        return gen_text.split('[/INST]')[-1].replace('</s>', '').strip()
    

class GenModelLlama3Instruct(GenModelBase):
    def __init__(self):
        self.model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        super().__init__()

    def __str__(self):
        return f'GenModelLlama3Instruct - {self.model_name}'

    def max_context_len(self):
        return 6000
    
    def parse_output(self, gen_text):
        return gen_text.split('<|end_header_id|>')[-1].replace('<|eot_id|>', '').strip()
    

class GenModelLlama3_1Instruct(GenModelBase):
    def __init__(self):
        self.model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        super().__init__()
        
    def __str__(self):
        return f'GenModelLlama3_1Instruct - {self.model_name}'

    def max_context_len(self):
        return 90000

    def parse_output(self, gen_text):
        return gen_text.split('<|end_header_id|>')[-1].replace('<|eot_id|>', '').strip()
    

class GenModelGPT4oMini(GenModelBase):
    def __init__(self):
        self.model_name = "gpt-4o-mini"
        super().__init__()

    def __str__(self):
        return f'GenModelGPT4oMini - {self.model_name}'

    def max_context_len(self):
        return 128000

    def load_model(self):
        self.client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY, 
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_API_ENDPOINT
        )
        logging.info(f'Loading model {self.model_name}')
    
    def generate(self, input_str):
        messages = [{"role": "user", "content": input_str}]
        num_tries = 1
        max_tries = 3
        while num_tries <= max_tries:  # try upto three times to get a proper response
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages
                )
                #tokens_in = response.usage.prompt_tokens
                #tokens_out = response.usage.completion_tokens
                response_content = response.choices[0].message.content
                break  # successful
            except Exception as ex:
                if num_tries < max_tries:
                    print(f'Something went wrong in try number {num_tries}. Retrying...')
                    num_tries += 1
                else:
                    print(f'Something went wrong {max_tries} times. No more tries!')
                    response_content = ''
                    break
        return response_content


if __name__ == '__main__':
    model = GenModelMistral7BInstruct()
    #model = GenModelLlama3Instruct()
    #model = GenModelLlama3_1Instruct()
    #model = GenModelGPT4oMini()
    inputs = ['Hello!', 'Given a text, tell me its sentiment.\nThe text is:\nThis is great']
    for i in inputs:
        output = model.generate(i)
        print('----------')
        print(f'INPUT:\n{i}')
        print('>>>>>>>>>>>')
        print(f'OUTPUT:\n{output}')
        print('<<<<<<<<<<<')