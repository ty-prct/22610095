import torch, time, psutil
from transformers import AutoModel, AutoTokenizer
import torch.nn.utils.prune as prune

model_name = "distilbert-base-uncased"
device = "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

def measure(model):
    inp = tokenizer("AI optimizes traffic flow.", return_tensors="pt").to(device)
    t0 = time.time()
    with torch.no_grad(): model(**inp)
    return (time.time() - t0) * 1000  # ms

print(f"Base latency: {measure(model):.2f} ms, Mem: {psutil.Process().memory_info().rss/1e6:.1f} MB")

# --- Pruning 30% weights ---
for _, m in model.named_modules():
    if isinstance(m, torch.nn.Linear):
        prune.l1_unstructured(m, 'weight', amount=0.3)
        prune.remove(m, 'weight')
print(f"After pruning latency: {measure(model):.2f} ms")

# --- Quantization ---
q_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
print(f"After quantization latency: {measure(q_model):.2f} ms")
print("Memory (MB):", psutil.Process().memory_info().rss/1e6)
