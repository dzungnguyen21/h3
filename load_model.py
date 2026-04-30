import gc
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

if "model" in globals():
    del model
gc.collect()
torch.cuda.empty_cache()

model_id = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="eager",
)
processor = AutoProcessor.from_pretrained(model_id)