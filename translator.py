"""
translator.py  –  MIT License
OCR (Tesseract 5.4), language detection (fastText),
translation (LibreTranslate OSS), PDF reconstruction (PyMuPDF).
"""

import os
import pathlib
import requests
import pytesseract
import pymupdf as fitz
from PIL import Image
import io
import json
import tqdm

# ----------------------------------------------------------------------
# 1.  Download large files only when first needed
# ----------------------------------------------------------------------
FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
FASTTEXT_LOCAL = pathlib.Path(__file__).parent / "lid.176.ftz"

def get_fasttext_model():
    """Return a fastText language-ID model (lazy download)."""
    if not FASTTEXT_LOCAL.exists():
        FASTTEXT_LOCAL.write_bytes(requests.get(FASTTEXT_URL, timeout=60).content)
    import fasttext
    return fasttext.load_model(str(FASTTEXT_LOCAL))

# ----------------------------------------------------------------------
# 2.  Helpers
# ----------------------------------------------------------------------
LIBRE_URL = "https://libretranslate.com/translate"   # free, no key

def tess_lang(lang_code: str) -> str:
    """Map human name to Tesseract script identifier."""
    mapping = {
        "Hindi": "Devanagari",
        "Bengali": "Bengali",
        "Tamil": "Tamil",
        "Telugu": "Telugu",
        "Marathi": "Devanagari",
        "Gujarati": "Gujarati",
        "Kannada": "Kannada",
        "Malayalam": "Malayalam",
        "Odia": "Oriya",
        "Punjabi": "Gurmukhi",
        "Assamese": "Bengali",
        "Urdu": "Arabic",
        "Sanskrit": "Devanagari",
        "Nepali": "Devanagari",
        "Konkani": "Devanagari",
        "Bodo": "Devanagari",
        "Dogri": "Devanagari",
        "Maithili": "Devanagari",
        "Manipuri": "Bengali",
        "Santhali": "Bengali",
        "Sindhi": "Arabic",
        "Kashmiri": "Arabic",
    }
    return mapping.get(lang_code, "Devanagari")

def detect_language(texts):
    """Return the dominant language code (e.g. 'en', 'hi')."""
    model = get_fasttext_model()
    joined = " ".join(texts).replace("\n", " ")
    pred = model.predict(joined, k=1)
    lang = pred[0][0].replace("__label__", "")
    # tiny map fastText code → human name
    code2lang = {"en": "English", "hi": "Hindi", "bn": "Bengali",
                 "ta": "Tamil", "te": "Telugu", "mr": "Marathi",
                 "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam",
                 "or": "Odia", "pa": "Punjabi", "as": "Assamese",
                 "ur": "Urdu", "sa": "Sanskrit", "ne": "Nepali",
                 "kok": "Konkani", "brx": "Bodo", "doi": "Dogri",
                 "mai": "Maithili", "mni": "Manipuri", "sat": "Santhali",
                 "sd": "Sindhi", "ks": "Kashmiri"}
    return code2lang.get(lang, lang)

def translate(text, source, target):
    """Translate via LibreTranslate public endpoint (free)."""
    if source == target or not text.strip():
        return text
    payload = {"q": text, "source": source, "target": target, "format": "text"}
    try:
        r = requests.post(LIBRE_URL, data=payload, timeout=30)
        r.raise_for_status()
        return r.json()["translatedText"]
    except Exception:
        return text  # graceful fallback

# ----------------------------------------------------------------------
# 3.  Main routine
# ----------------------------------------------------------------------
def process_pdf(src_path, dst_path, target_lang=None):
    doc = fitz.open(src_path)
    all_texts = []

    # 1. Extract / OCR text per page
    for page in tqdm.tqdm(doc, desc="OCR"):
        if page.get_text().strip():
            all_texts.append(page.get_text())
        else:  # scanned image
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            img = Image.open(io.BytesIO(pix.tobytes()))
            lang = detect_language([page.get_text()])
            tess_lang_code = tess_lang(lang)
            text = pytesseract.image_to_string(
                img,
                lang=tess_lang_code,
                config="--tessdata-dir /usr/share/tesseract-ocr/4.00/tessdata"
            )
            all_texts.append(text)

    detected_lang = detect_language(all_texts)
    if target_lang is None:
        target_lang = "English"
    translated_pages = [translate(t, detected_lang, target_lang) for t in all_texts]

    # 2. Re-assemble PDF
    out = fitz.open()
    for idx, (page, new_text) in enumerate(zip(doc, translated_pages)):
        rect = page.rect
        new_page = out.new_page(width=rect.width, height=rect.height)

        if page.get_text().strip():  # native PDF → replace text
            new_page.insert_text(
                fitz.Point(72, 72), new_text, fontname="helv", fontsize=11
            )
        else:  # scanned → overlay
            new_page.insert_text(
                fitz.Point(72, 72), new_text, fontname="helv", fontsize=11, color=(1, 0, 0)
            )
        # copy original images
        for img in page.get_images(full=True):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            new_page.insert_image(rect, pixmap=pix)

    out.save(dst_path)
