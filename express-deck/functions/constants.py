import torch


BACKGROUNDS = {
    "style1.css": ["./templates/backgrounds/bg_img_style10.png","./templates/backgrounds/bg_img_style11.png","./templates/backgrounds/bg_img_style12.png","./templates/backgrounds/bg_img_style13.png"],
    "style2.css": ["./templates/backgrounds/bg_img_style20.png","./templates/backgrounds/bg_img_style21.png","./templates/backgrounds/bg_img_style22.png","./templates/backgrounds/bg_img_style23.png"],
    "style3.css": ["./templates/backgrounds/bg_img_style30.png","./templates/backgrounds/bg_img_style31.png","./templates/backgrounds/bg_img_style32.png","./templates/backgrounds/bg_img_style33.png"]
}


NEG_PROMPT = """
"UnrealisticDream, FastNegativeEmbedding,worst quality, low quality:1.4, blurry:1.2,blurry face,
blurry eyes,greyscale, monochrome:1.1, 3D face, cropped, text, jpeg artifacts, watermark, username,
blurry, artist name, watermark, title, multiple view, , plump, fat, muscular female, strabismus, out of frame,
ugly, extra limbs, bad anatomy,p gross proportions, malformed limbs, missing arms, missing legs,
extra arms, extra legs, mutated hands, fused fingers, distorted face, too many fingers,
long neck"
"""

DEVICE = "cuda" if torch.cuda.is_available else "cpu"
