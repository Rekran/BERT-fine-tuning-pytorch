import argparse
import bitsandbytes as bnb
from datasets import load_dataset
from functools import partial
import os
from datasets import load_dataset
from model import *

# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
# from transformers import set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, \
#     DataCollatorForLanguageModeling, Trainer, TrainingArguments


# model_name = "google-bert/bert-base-uncased" 
model_name = "IMSyPP/hate_speech_en"

bnb_config = create_bnb_config()

print(bnb_config)

model, tokenizer = load_model(model_name, bnb_config)

