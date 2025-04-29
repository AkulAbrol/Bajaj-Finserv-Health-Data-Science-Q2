# File: main.py

import os
import io
import re
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from base64 import b64encode
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageEnhance
import pytesseract

# =======================
# Configuration
# =======================
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

STATIC_DIR = Path("static")
TEMPLATE_HTML = Path("templates/index.html")

# =======================
# Image Processing
# =======================
def preprocess_image(pil_image: Image.Image) -> Image.Image:
    img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(thresh)

# =======================
# OCR and Parsing
# =======================
def extract_text_from_image(image: Image.Image) -> str:
    return pytesseract.image_to_string(image)

def check_out_of_range(value: str, ref_range: str) -> bool:
    try:
        value = float(value)
        numbers = [float(n) for n in re.findall(r"[-+]?\d*\.?\d+", ref_range)]
        if len(numbers) == 2:
            return value < numbers[0] or value > numbers[1]
    except:
        return False
    return False

def parse_lab_test_results(text: str) -> dict:
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    results = []
    pattern = re.compile(r'^(.*?)([:\-]?)\s*([\d.]+)\s*(\[\w*\])?\s*([a-zA-Z/%\u03bc\u00b5ulUL\d]+)?\s*([<>]?=?\s*[\d.]+[-\u2013\u2014\u2212]\s*[\d.]+)?')

    for line in lines:
        match = pattern.match(line)
        if match:
            test_name = match.group(1).strip(" :-")
            test_value = match.group(3)
            unit = match.group(5) or ""
            ref_range = match.group(6) or ""
            out_of_range = check_out_of_range(test_value, ref_range)

            results.append({
                "test_name": test_name,
                "test_value": test_value,
                "test_unit": unit,
                "bio_reference_range": ref_range,
                "lab_test_out_of_range": out_of_range
            })

    return {"is_success": True, "data": results}

# =======================
# FastAPI App
# =======================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(content=TEMPLATE_HTML.read_text())

@app.post("/get-lab-tests")
async def get_lab_tests(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        processed_image = preprocess_image(image)
        text = extract_text_from_image(processed_image)
        results = parse_lab_test_results(text)

        buffered = io.BytesIO()
        processed_image.save(buffered, format="PNG")
        img_str = b64encode(buffered.getvalue()).decode()
        results["enhanced_image"] = f"data:image/png;base64,{img_str}"

        return JSONResponse(content=results)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# =======================
# Local Testing Hook
# =======================
if __name__ == "__main__":
    import uvicorn
    test_image = r"C:\\Users\\DELL\\OneDrive\\Desktop\\bajaj\\lab_reports_samples\\lbmaske\\AHD-0425-PA-0008061_E-mahendrasinghdischargecard_250427_1114@E.pdf_page_27.png"
    try:
        with Image.open(test_image) as img:
            processed = preprocess_image(img)
            text = extract_text_from_image(processed)
            results = parse_lab_test_results(text)
            df = pd.DataFrame(results["data"])
            print(df)
    except Exception as e:
        print("Error:", e)

    uvicorn.run(app, host="127.0.0.1", port=8000)
