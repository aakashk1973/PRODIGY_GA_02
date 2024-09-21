# Install the necessary packages first 
!pip install torch torchvision torchaudio diffusers

from diffusers import StableDiffusionPipeline
import torch

# Load the model
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id)

# running the model on CPU
pipe.to("cpu")

# Generating an image based on a prompt
prompt = "A futuristic city skyline at sunset"
image = pipe(prompt).images[0]

# Saving the generated image
image.save("generated_image.png")
