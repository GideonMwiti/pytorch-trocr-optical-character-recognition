"""
Synthetic OCR image generation script for TrOCR benchmark evaluation.
Creates 8 distinct text line images mimicking diverse real-world documents,
receipts, signage, code, and book extracts.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(DATA_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

def render_text_image(
    text: str,
    output_path: str,
    font_size: int = 36,
    bg_color: tuple = (255, 255, 255),
    text_color: tuple = (0, 0, 0),
    padding: tuple = (24, 18),
    add_noise: bool = False,
    italic: bool = False
):
    """Renders a text string onto a clean or noisy cropped document patch."""
    # Attempt to load a clean TTF font or fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    # Determine bounding box
    dummy_img = Image.new("RGB", (10, 10))
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Create image with padding
    pad_x, pad_y = padding
    img_w = max(text_w + 2 * pad_x, 384)
    img_h = text_h + 2 * pad_y
    img = Image.new("RGB", (img_w, img_h), bg_color)
    draw = ImageDraw.Draw(img)

    # Center text horizontally if padded wide
    pos_x = (img_w - text_w) // 2
    pos_y = (img_h - text_h) // 2 - bbox[1]

    draw.text((pos_x, pos_y), text, fill=text_color, font=font)

    if add_noise:
        # Add subtle pepper noise and Gaussian blur to simulate degraded scanning
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, 12, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=0.6))

    img.save(output_path, quality=95)


def generate_all_samples():
    samples = [
        {
            "filename": "sample_quote.png",
            "text": "Artificial Intelligence is transforming computing.",
            "font_size": 34,
            "bg": (255, 255, 255),
            "fg": (20, 20, 20),
            "noise": False
        },
        {
            "filename": "sample_receipt.png",
            "text": "TOTAL AMOUNT DUE: $48.95",
            "font_size": 36,
            "bg": (248, 248, 240),
            "fg": (30, 30, 30),
            "noise": False
        },
        {
            "filename": "sample_pangram.png",
            "text": "The quick brown fox jumps over the lazy dog.",
            "font_size": 32,
            "bg": (255, 255, 255),
            "fg": (10, 10, 10),
            "noise": False
        },
        {
            "filename": "sample_code.png",
            "text": "import torch.nn as nn",
            "font_size": 36,
            "bg": (250, 250, 252),
            "fg": (25, 40, 80),
            "noise": False
        },
        {
            "filename": "sample_invoice.png",
            "text": "INVOICE #94820 DATE: 2026-11-02",
            "font_size": 34,
            "bg": (255, 255, 255),
            "fg": (15, 15, 15),
            "noise": False
        },
        {
            "filename": "sample_sign.png",
            "text": "SPEED LIMIT 35 MPH",
            "font_size": 38,
            "bg": (255, 255, 255),
            "fg": (0, 0, 0),
            "noise": False
        },
        {
            "filename": "sample_medical.png",
            "text": "DOSAGE: TAKE 2 TABLETS DAILY",
            "font_size": 34,
            "bg": (245, 248, 250),
            "fg": (20, 30, 40),
            "noise": False
        },
        {
            "filename": "sample_degraded.png",
            "text": "PYTORCH DEEP LEARNING FRAMEWORK",
            "font_size": 34,
            "bg": (240, 240, 240),
            "fg": (40, 40, 40),
            "noise": True
        }
    ]

    for s in samples:
        out_p = os.path.join(IMAGES_DIR, s["filename"])
        render_text_image(
            text=s["text"],
            output_path=out_p,
            font_size=s["font_size"],
            bg_color=s["bg"],
            text_color=s["fg"],
            add_noise=s["noise"]
        )

    print(f"Generated {len(samples)} synthetic OCR benchmark images in: {IMAGES_DIR}")


if __name__ == "__main__":
    generate_all_samples()

