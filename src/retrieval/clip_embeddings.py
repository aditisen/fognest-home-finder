from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from ..config import settings


@lru_cache(maxsize=1)
def _load_clip():
    """Load CLIP model and processor (cached). Downloads ~400 MB on first run."""
    print(f"  Loading CLIP model {settings.clip_model_name} ...")
    model = CLIPModel.from_pretrained(settings.clip_model_name)
    processor = CLIPProcessor.from_pretrained(settings.clip_model_name)
    model.eval()
    return model, processor


def embed_image(image_path: Path) -> list[float]:
    """Return a normalized CLIP image embedding (512-dim)."""
    model, processor = _load_clip()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        vision_out = model.vision_model(pixel_values=inputs["pixel_values"])
        features = model.visual_projection(vision_out.pooler_output)
        features = torch.nn.functional.normalize(features, p=2, dim=-1)
    return features.squeeze().tolist()


def embed_text(text: str) -> list[float]:
    """
    Return a normalized CLIP text embedding (512-dim).
    Lives in the same vector space as embed_image(), enabling text-to-photo search.
    """
    model, processor = _load_clip()
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_out = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        features = model.text_projection(text_out.pooler_output)
        features = torch.nn.functional.normalize(features, p=2, dim=-1)
    return features.squeeze().tolist()