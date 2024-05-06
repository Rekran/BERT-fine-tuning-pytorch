import torch
from transformers import BertTokenizer
import numpy as np 
import pandas as pd 

print('Loading BERT tokenizer...')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

print(tokenizer.tokenize("Hello, my dog is cute."), tokenizer.convert_tokens_to_ids(tokenizer.tokenize("Hello, my dog is cute.")))



