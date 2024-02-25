from fastapi import FastAPI
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
app = FastAPI()


processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

@app.get("/")
def root():
    return {"message": "Hello World"}

@app.get("/add")
def add_numbers(num1: float, num2: float):
    result = num1 + num2
    return {"result": result}


@app.get("/complete")
def complete_sentence(sent:str):
    result = "Hello "  + sent
    return {"result": result}



@app.get("/predict_image")
def add_numbers(img_url:str):
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs)
    result = processor.decode(out[0], skip_special_tokens=True)
    return {"result": result}


# img_url = '' 
