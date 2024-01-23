from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import uvicorn
import transformers
import os
import torch
import cloudinary
import cloudinary.uploader
import cloudinary.api
from PIL import Image
from io import BytesIO
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from diffusers import StableDiffusionPipeline

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cloudinary.config(
    cloud_name = "du3z4clzd",
    api_key = "553277872935676",
    api_secret = "gcsWnw9N_05VyxYFtlNbwvwjihE",
)

class Prompt(BaseModel):
    prompt_text: str
    lang: str

model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')
tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')
pipe = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16)
pipe = pipe.to("cuda")


language = {
    'German' : 'de_DE',
    'English': 'en_XX',
    'Spanish': 'es_XX',
    'French': 'fr_XX',
    'Gujarati': 'gu_IN',
    'Hindi': 'hi_IN',
    'Japanese': 'ja_XX',
    'Dutch': 'nl_XX'
}

def translator(input_text, lang):
  tokenizer.src_lang = lang
  encoded_hi = tokenizer(input_text, return_tensors="pt")
  generated_tokens = model.generate(
      **encoded_hi,
      forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
  )
  return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]


def generate_image(prompt):
  image_path = f'../generated_images/{prompt}.png'
  image = pipe(prompt).images[0]
  image.save(image_path)

  pil_image = Image.open(image_path)
  image_bytes = BytesIO()
  pil_image.save(image_bytes, format='PNG')
  image_bytes = image_bytes.getvalue()

  imgUrl = cloudinary.uploader.upload(image_bytes)
  return imgUrl['url']

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

@app.post('/predict')
async def predict(input_prompt: Prompt):
    prompt_text = input_prompt.prompt_text
    lang = language[input_prompt.lang]
    
    translated_prompt = translator(prompt_text, lang)
    imgurl = generate_image(translated_prompt)
    
    return{ 
        "image": imgurl
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
