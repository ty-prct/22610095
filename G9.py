from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss, numpy as np, torch

# Small doc set
docs = ["Paris is capital of France.", "Python is used in data science.", "Taj Mahal is in Agra, India."]
emb = SentenceTransformer("all-MiniLM-L6-v2")
vecs = emb.encode(docs).astype("float32"); faiss.normalize_L2(vecs)
index = faiss.IndexFlatIP(vecs.shape[1]); index.add(vecs)
gen = pipeline("text2text-generation", model="google/flan-t5-small", device=0 if torch.cuda.is_available() else -1)

def rag(q):
    qv = emb.encode([q]).astype("float32"); faiss.normalize_L2(qv)
    _, I = index.search(qv, 2)
    ctx = " ".join(docs[i] for i in I[0])
    return gen(f"Context: {ctx}\nQuestion: {q}\nAnswer:", max_length=40)[0]['generated_text']

for q in ["Where is Taj Mahal?", "What is Python used for?"]:
    print(f"\nQ: {q}\nA: {rag(q)}")
