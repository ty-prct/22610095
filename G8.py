from transformers import pipeline
from evaluate import load

# --- Load pre-trained LLMs (CPU only) ---
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2", device=-1)

# --- Prompts ---
text = """Artificial Intelligence (AI) is transforming industries by automating tasks,
enhancing decision-making, and creating new opportunities across sectors such as healthcare,
finance, and education."""
question = "How is AI transforming industries?"

# --- Outputs ---
summary = summarizer(text, max_length=50, min_length=20, do_sample=False)[0]['summary_text']
answer = qa_model(question=question, context=text)['answer']

print("Summary:\n", summary)
print("\nAnswer:\n", answer)

# --- Evaluation with ROUGE ---
rouge = load("rouge")
ref_summary = "AI automates tasks and improves decision-making in various sectors."
results = rouge.compute(predictions=[summary], references=[ref_summary])
print("\nROUGE Scores:", results)
