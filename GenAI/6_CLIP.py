import torch 
from diffusers import StableDiffusionPipeline 

def generate_simple_image(prompt, output_path="output.png", steps=30): 
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16 if device=="cuda" else torch.float32, safety_checker=None) 
    pipe = pipe.to(device) 
    image = pipe(prompt, num_inference_steps=steps).images[0] 
    image.save(output_path) 

if __name__ == "__main__": 
    prompt = "an astronaut bear on mars" 
    generate_simple_image(prompt) 