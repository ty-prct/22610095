# Fine-Tuning a Transformer-based Text Generation Model using LoRA
# Author: Utkarsh Pingale

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Load pre-trained DistilGPT-2
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Apply LoRA (Low-Rank Adaptation)
lora_cfg = LoraConfig(r=8, target_modules=["c_attn"])
model = get_peft_model(model, lora_cfg)

# Small custom text corpus
texts = ["AI makes life easier.", "Deep learning changes technology.", "LoRA helps fine-tune models fast!"]

# Fine-tune (single-step demo)
inputs = tokenizer("\n".join(texts), return_tensors="pt")
outputs = model(**inputs, labels=inputs["input_ids"])
loss = outputs.loss
loss.backward()  # single-step update

# Generate new text
inp = tokenizer("Artificial intelligence and deep learning", return_tensors="pt")
gen = model.generate(**inp, max_length=40)
print("\nGenerated Text:\n", tokenizer.decode(gen[0]))

# Evaluate Perplexity
import math
print("\nModel Perplexity:", round(math.exp(loss.item()), 2))