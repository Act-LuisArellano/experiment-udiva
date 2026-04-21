import torch
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("google/gemma-4-E2B", cache_dir="../data-slow/models/Gemma/current-model-variation")
print("Image token:", processor.image_token if hasattr(processor, "image_token") else "N/A")
print("Processor type:", type(processor))
if hasattr(processor, "tokenizer"):
    print("Has tokenizer")
    print("Tokens:", processor.tokenizer.convert_tokens_to_ids("<image>"))
