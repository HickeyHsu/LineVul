import pandas as pd
from transformers import AutoTokenizer
tokenizer_name=r"/home/hickey/python-workspace/LineVul/linevul/saved_models/razent/cotext-1-ccg"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
print(tokenizer.cls_token,tokenizer.sep_token,tokenizer.pad_token,tokenizer.bos_token,tokenizer.mask_token)
ids=tokenizer.convert_tokens_to_ids([tokenizer.bos_token,tokenizer.pad_token,tokenizer.mask_token])
print(ids)