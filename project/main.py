from model import *

# import bitsandbytes as bnb
# from datasets import load_dataset
# from functools import partial
# from datasets import load_dataset
# import argparse
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
# from transformers import set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, \
#     DataCollatorForLanguageModeling, Trainer, TrainingArguments


model_name = "neuralmind/bert-base-portuguese-cased"

bnb_config = create_bnb_config()

model = load_model(model_name, bnb_config)

tokenizer = load_tokenizer(model_name)

print(
tokenizer("Hello, world!")
)