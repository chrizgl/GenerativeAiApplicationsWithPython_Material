import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M", device_map="auto").eval()
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

# translate English to French
en_text = "Be the change you wish to see in the world."
tokenizer.src_lang = "en"
encoded_en = tokenizer(en_text, return_tensors="pt").to(model.device)
generated_tokens = model.generate(**encoded_en, forced_bos_token_id=tokenizer.get_lang_id("ja"))
translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print(f"Original: {en_text}")
print(f"Translation (Japanese): {translation[0]}")