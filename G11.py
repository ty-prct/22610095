from transformers import pipeline

# Load a small pre-trained generative model
gen = pipeline("text-generation", model="distilgpt2")

# Prompts to test bias
prompts = ["The nurse said that", "The engineer said that"]
for p in prompts:
    out = gen(p, max_length=15, num_return_sequences=3)
    texts = [o["generated_text"] for o in out]

    # --- Simple bias mitigation: remove gendered words ---
    filtered = [t.replace(" he", "").replace(" she", "").replace(" him", "").replace(" her", "") for t in texts]
    print(f"\nPrompt: {p}")
    print("Before:", texts)
    print("After :", filtered)

