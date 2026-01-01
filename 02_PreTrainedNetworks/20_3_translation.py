# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("translation", model="Mitsua/elan-mt-bt-en-ja")

# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Mitsua/elan-mt-bt-en-ja")
model = AutoModelForSeq2SeqLM.from_pretrained("Mitsua/elan-mt-bt-en-ja")

en_text = "Be the change you wish to see in the world."
inputs = tokenizer(en_text, return_tensors="pt")
outputs = model.generate(**inputs)
translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print(f"Original: {en_text}")
print(f"Pipeline Translation: {pipe(en_text)[0]['translation_text']}")
print(f"Direct Model Translation: {translation}")

# Original: Be the change you wish to see in the world.
# Pipeline Translation: 世の中の見たい変化になれ。
# Direct Model Translation: 世の中の見たい変化になれ。
# My comment: Best result