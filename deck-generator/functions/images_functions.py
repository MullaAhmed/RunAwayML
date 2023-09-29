from PIL import Image
from io import BytesIO
import requests,random,cv2,base64
from .constants import *
import numpy as np
from diffusers import DiffusionPipeline

def get_background(style):
    images=BACKGROUNDS[style]
    img=Image.open(random.choice(images))
    return img


def url_to_image(url) :

    images = [Image.open(BytesIO(requests.get(image_url).content)) for image_url in url]

    return images

def get_unsplash_img(query, orientation='portrait',  token= None,model=None) :

    api_key_unsplash = token if token else "xUtDrJibYywEGkH22JqoJ2Mc32iIFDVd_3767V6gNgM"
    
    base = "https://api.unsplash.com/search/photos/"
    request_query = f'{base}?query={query}&orientation={orientation}&per_page=10&client_id={api_key_unsplash}'
    response = requests.get(request_query)
    
    data = response.json()
    results = data['results'] 
    
    image_urls = [photo['urls']['full'] for photo in results]
   
    images = []
    for url in image_urls:
        image = url_to_image([url])
        if image:
            images.extend(image)

    random.shuffle(images)  # Shuffle the results for randomness
    return images[0]


def get_pexels_img(query, orientation='portrait',  token = None,model=None) :

    headers = {"Authorization": token if token else "9uDQSj9uz5lStKupRrb5hgnDK8cYcW1OHAJywoytwXgg9GRFGqukvetT"}
    request_query = f"https://api.pexels.com/v1/search?query={query}&orientation={orientation}&per_page=10"
    response = requests.get(request_query, headers=headers)
    
    data = response.json()
    results = data['photos']
    image_urls = [photo['src']['large2x'] for photo in results]
    
    images = []

    for url in image_urls:
        image = url_to_image([url])
        if image:
            images.extend(image)
    
    random.shuffle(images)  # Shuffle the results for randomness
    return images[0]

def generate_sd_image(prompt, orientation='portrait', token = None, model = "Lykon/DreamShaper"):

    pipeline = DiffusionPipeline.from_pretrained(model)
    pipeline.to(DEVICE)
   
    if orientation == "landscape":
        height = 600
        width = 824
    if orientation == "portrait":
        height = 824
        width = 600

    neg = NEG_PROMPT
    images = pipeline(
        prompt,
        guidance_scale=7.5,
        num_inference_steps=25,
        negative_prompt = neg ,
        
        height=height,
        width=width
    ).images

    return images[0]

def image_to_base64(image):

    header = "data:image/png;base64,"
    if isinstance(image, np.ndarray):
        _, image_encoded = cv2.imencode("png", image)
    else:
        with BytesIO() as buffer:
            image.save(buffer, format="png")
            buffer.seek(0)
            image_encoded = buffer.getvalue()
            buffer.truncate(0)

    base64_string = base64.b64encode(image_encoded).decode("utf-8")

    return f"{header}{base64_string}"

