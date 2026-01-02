from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torch, scipy.io.wavfile as wav
import numpy as np
import os

device = 0 if torch.cuda.is_available() else -1
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained(
    "facebook/musicgen-small",
    attn_implementation="eager",  # vermeidet die SDPA-Warnung
    torch_dtype=torch.float16,
).to(device)

inputs = processor(
    text=["80s pop track with bassy drums and synth"],
    padding=True,
    return_tensors="pt",
).to(device)

# Tokens generieren (nicht Logits speichern)
gen_tokens = model.generate(
    **inputs,
    do_sample=True,
    temperature=1.0,
    top_k=250,
    top_p=0.95,
    max_new_tokens=1024,  # ca. ~30-60s; wenn 0s rauskommt, auf 768-1024 erhöhen
)

# Tokens -> Audio dekodieren
audio_values = processor.batch_decode(gen_tokens, sampling_rate=32000)  # Liste von np.array
audio = audio_values[0]  # erstes Sample

# Falls leer, Hinweis geben
if audio.size == 0:
    raise RuntimeError("Decodiertes Audio ist leer. Erhöhe max_new_tokens (z.B. 768 oder 1024) oder prüfe die Prompt-Eingabe.")

# Debug: Prüfe Form und Pegel
print(f"Audio shape: {audio.shape}, dtype: {audio.dtype}, min: {audio.min()}, max: {audio.max()}")
if np.allclose(audio, 0):
    raise RuntimeError("Decodiertes Audio besteht nur aus Nullen (Stille). Erhöhe max_new_tokens oder versuche ohne float16.")

# Falls mehrfach Kanäle oder Extra-Dimensionen: auf Mono reduzieren
if audio.ndim > 2:
    audio = audio.squeeze()
if audio.ndim == 2:
    if audio.shape[0] <= 4:  # (channels, samples)
        audio = audio[0]
    else:  # (samples, channels)
        audio = audio[:, 0]

print(f"After channel selection: {audio.shape}, min: {audio.min()}, max: {audio.max()}")

# WAV speichern (int16)
audio = np.clip(audio.astype(np.float32), -1.0, 1.0)
audio_int16 = (audio * 32767).astype("int16")
wav.write("musicgen_small_out.wav", 32000, audio_int16)

# Dauer prüfen und melden
duration_sec = audio_int16.shape[0] / 32000.0
file_size = os.path.getsize("musicgen_small_out.wav")
print(f"Saved musicgen_small_out.wav | duration ~{duration_sec:.2f}s | bytes {file_size}")