from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt

model_id = "CompVis/stable-diffusion-v1-4"
# model_id = "runwayml/stable-diffusion-v1-5"

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

prompt = "A fantasy landscape, trending on artstation"

image = pipe(prompt).images[0]

plt.imshow(image)
plt.axis("off")
plt.show()