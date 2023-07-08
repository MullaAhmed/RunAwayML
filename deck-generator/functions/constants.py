import torch


BACKGROUNDS = {
    "style1.css": "./templates/backgrounds/bg_img_style1.png",
    "style2.css": "./templates/backgrounds/bg_img_style2.png",
    "style3.css": "./templates/backgrounds/bg_img_style3.png",
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
