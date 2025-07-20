"""
app.py  –  MIT License
Single-file Streamlit UI for the 22-script Indic PDF translator.
"""

import streamlit as st
from translator import process_pdf
from pathlib import Path
import shutil
import tempfile
import os

st.set_page_config(
    page_title="Indic PDF Translator (100 % Free)",
    page_icon="📄",
    layout="centered",
)

st.title("📄 Free Indic PDF Translator")
st.markdown(
    "Upload **any** Indian-language PDF (scanned or native). "
    "We auto-detect the source language, translate, and give you back a new PDF."
)

uploaded = st.file_uploader("Choose PDF", type=["pdf"])
if uploaded:
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "input.pdf"
        out_path = Path(tmpdir) / "translated.pdf"
        with open(src_path, "wb") as f:
            f.write(uploaded.read())

        # ---- Language selection ----
        indic_plus = [
            "Auto-detect",
            "Hindi",
            "Bengali",
            "Tamil",
            "Telugu",
            "Marathi",
            "Gujarati",
            "Kannada",
            "Malayalam",
            "Odia",
            "Punjabi",
            "Assamese",
            "Urdu",
            "Sanskrit",
            "Nepali",
            "Konkani",
            "Bodo",
            "Dogri",
            "Maithili",
            "Manipuri",
            "Santhali",
            "Sindhi",
            "Kashmiri",
            "English",
            "Spanish",
            "French",
            "Arabic",
            "Chinese",
            "Russian",
        ]
        tgt = st.selectbox("Translate into:", indic_plus)

        if st.button("Translate PDF"):
            with st.spinner("OCR → Detect language → Translate → Re-assemble PDF …"):
                try:
                    process_pdf(
                        src_path,
                        out_path,
                        target_lang=None if tgt == "Auto-detect" else tgt,
                    )
                    st.success("Done!")
                    with open(out_path, "rb") as f:
                        st.download_button(
                            label="⬇️  Download Translated PDF",
                            data=f.read(),
                            file_name=f"translated_{uploaded.name}",
                            mime="application/pdf",
                        )
                except Exception as e:
                    st.error(str(e))
