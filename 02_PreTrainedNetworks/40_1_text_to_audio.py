# Use a pipeline as a high-level helper
from transformers import pipeline
import scipy.io.wavfile as wav
import torch
import numpy as np

# Check for CUDA
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'CUDA' if device == 0 else 'CPU'}")

# Create pipeline with memory optimization for small model (fast + fits RTX 3070)
pipe = pipeline(
    "text-to-audio",
    model="facebook/musicgen-small",
    device=device,
    torch_dtype=torch.float16,
    model_kwargs={"attn_implementation": "eager"},  # avoid fallback warning
)

# Sampling tuned for etwas mehr Fülle ohne Instabilität
output = pipe(
    "80s synthesizer dance music instrumental no vocals",
    do_sample=True,
    temperature=1.0,
    top_k=250,
    top_p=0.95,
)

# Save the audio to a file
# Output format: {"audio": array, "sampling_rate": int}
sampling_rate = output["sampling_rate"]
audio_data = output["audio"]

# Ensure audio is 1D or 2D and squeeze if needed
if audio_data.ndim > 2:
    audio_data = audio_data.squeeze()
if audio_data.ndim == 2:
    audio_data = audio_data[0]  # Get first channel if stereo

print(f"Audio shape: {audio_data.shape}, dtype: {audio_data.dtype}, min: {audio_data.min()}, max: {audio_data.max()}")

# Convert float to int16 safely
audio_data = np.clip(audio_data, -1.0, 1.0)  # Ensure in range
audio_int16 = np.int16(audio_data * 32767)

# Save as WAV file
wav.write("generated_music.wav", rate=int(sampling_rate), data=audio_int16)
print("Audio saved as 'generated_music.wav'")

# Optional: Play directly in Jupyter/IPython
# from IPython.display import Audio
# Audio(audio_data, rate=sampling_rate)

