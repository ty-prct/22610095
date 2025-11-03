# Analyze bias in a pre-trained generative model (DistilGPT2)
# Mitigation: Simple gender-word filtering
from transformers import pipeline

gen = pipeline("text-generation", model="distilgpt2")
prompts = ["The nurse said that", "The engineer said that"]

for p in prompts:
    out = gen(p, max_length=15, num_return_sequences=2)
    texts = [o["generated_text"] for o in out]

    # Simple bias mitigation: remove gendered words
    filtered = [t.replace(" he", "").replace(" she", "").replace(" him", "").replace(" her", "") for t in texts]

    print(f"\nPrompt: {p}")
    print("Before:", texts)
    print("After :", filtered)
