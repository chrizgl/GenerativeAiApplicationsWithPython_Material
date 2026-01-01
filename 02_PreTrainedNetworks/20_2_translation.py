# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-jap")

# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-jap")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-jap")
en_text = "Be the change you wish to see in the world."
inputs = tokenizer(en_text, return_tensors="pt")
outputs = model.generate(**inputs)
translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

print(f"Original: {en_text}")
print(f"Pipeline Translation: {pipe(en_text)[0]['translation_text']}")
print(f"Direct Model Translation: {translation}")