"""
translator.py  –  MIT License
OCR (Tesseract 5.4 best), language detection (fastText),
translation (LibreTranslate OSS), PDF reconstruction (PyMuPDF).
"""

import os
import pymupdf as fitz   # PyMuPDF ≥ 1.23
import pytesseract
from PIL import Image
import io
import requests
import fasttext
import pathlib
import json
import tqdm

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
TESSDATA = pathlib.Path(__file__).parent / "tessdata_best"
FASTTEXT_MODEL = pathlib.Path(__file__).parent / "lid.176.bin"
LIBRE_URL = "http://libretranslate:5000/translate"  # default local endpoint

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def get_fasttext_model():
    if not FASTTEXT_MODEL.exists():
        url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
        pathlib.Path(FASTTEXT_MODEL).write_bytes(requests.get(url).content)
    return fasttext.load_model(str(FASTTEXT_MODEL))

def tess_lang(lang_code):
    """
    Map human names → tesseract script names.
    Add more mappings as required.
    """
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
        "Santhali": "Bengali",  # Ol-Chiki not in tesseract
        "Sindhi": "Arabic",
        "Kashmiri": "Arabic",
    }
    return mapping.get(lang_code, "Devanagari")

def detect_language(texts):
    model = get_fasttext_model()
    joined = " ".join(texts)
    pred = model.predict(joined.replace("\n", " "), k=1)
    lang = pred[0][0].replace("__label__", "")
    # Map fastText code to human name
    code2lang = json.loads(
        pathlib.Path(__file__).parent.joinpath("lang_codes.json").read_text()
    )
    return code2lang.get(lang, lang)

def translate(text, source, target):
    """
    Uses a local LibreTranslate container; falls back to Google-free
    if LIBRE_URL is unreachable (prints warning).
    """
    if source == target or not text.strip():
        return text
    payload = {
        "q": text,
        "source": source,
        "target": target,
        "format": "text",
    }
    try:
        r = requests.post(LIBRE_URL, data=payload, timeout=30)
        r.raise_for_status()
        return r.json()["translatedText"]
    except Exception as e:
        print("LibreTranslate unavailable:", e)
        return text  # graceful fallback

# ------------------------------------------------------------------
# Main routine
# ------------------------------------------------------------------
def process_pdf(src_path, dst_path, target_lang=None):
    doc = fitz.open(src_path)
    all_texts = []

    # 1. Collect text (native or OCR)
    for page in tqdm.tqdm(doc, desc="OCR"):
        if page.get_text().strip():  # native PDF
            all_texts.append(page.get_text())
        else:  # scanned image
            mat = fitz.Matrix(2, 2)  # 2× DPI
            pix = page.get_pixmap(matrix=mat)
            img = Image.open(io.BytesIO(pix.tobytes()))
            # Detect script per page (cheap heuristic)
            lang_hint = detect_language([page.get_text()])
            tess_lang_code = tess_lang(lang_hint)
            text = pytesseract.image_to_string(
                img,
                lang=tess_lang_code,
                config=f"--tessdata-dir {TESSDATA}",
            )
            all_texts.append(text)

    detected_lang = detect_language(all_texts)
    if target_lang is None:
        target_lang = "English"  # default when auto-detect chosen

    # 2. Translate page-by-page
    translated_pages = [translate(t, detected_lang, target_lang) for t in all_texts]

    # 3. Re-assemble PDF
    out = fitz.open()
    for idx, (page, new_text) in enumerate(zip(doc, translated_pages)):
        rect = page.rect
        new_page = out.new_page(width=rect.width, height=rect.height)

        if page.get_text().strip():  # native → replace text
            new_page.insert_text(
                fitz.Point(72, 72),
                new_text,
                fontname="helv",
                fontsize=11,
                rotate=0,
            )
        else:  # scanned → overlay
            new_page.insert_text(
                fitz.Point(72, 72),
                new_text,
                fontname="helv",
                fontsize=11,
                rotate=0,
                color=(1, 0, 0),
            )
            # TODO: keep images + overlay exact bounding boxes
        # Copy original images
        for img in page.get_images(full=True):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            new_page.insert_image(rect, pixmap=pix)

    out.save(dst_path)
