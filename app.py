from flask import Flask, render_template, request, jsonify
import torch
from diffusers import StableDiffusionPipeline

app = Flask(__name__)

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
else:
    pipe = StableDiffusionPipeline.from_pretrained(model_id)

pipe = pipe.to(device)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form.get('prompt')
    
    try:
        # Generate the image
        image = pipe(prompt, guidance_scale=7.5, height=512, width=512).images[0]
        
        # Save the generated image
        image_path = "static/generated_img.png"
        image.save(image_path)

        return jsonify({'image_url': image_path})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
