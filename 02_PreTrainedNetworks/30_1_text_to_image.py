import torch
from diffusers import AmusedPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = AmusedPipeline.from_pretrained(
    "amused/amused-256",
    variant="fp16",
    torch_dtype=torch.float16,
).to(device)

# vqvae in fp32, aber weiterhin auf GPU
pipe.vqvae.to(device=device, dtype=torch.float32)

prompt = "Ein rotes Haus"

gen = torch.Generator(device=device).manual_seed(8)

image = pipe(prompt, generator=gen).images[0]
image.save("text2image_256.png")
