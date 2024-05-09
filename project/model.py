from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv(dotenv_path = '../.env')
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')


def load_model(model_name, bnb_config):
    n_gpus = torch.cuda.device_count()
    max_memory = f'{4000}MB'

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        max_memory = {i: max_memory for i in range(n_gpus)},
        device_map="cuda",
        low_cpu_mem_usage=True,
        is_decoder=True,
        
    )
    return model

def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token = HUGGINGFACE_TOKEN)  
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config