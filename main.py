#!/usr/bin/env python
# coding: utf-8

# In[22]:


import os
import re
from PIL import Image, ImageEnhance
import pytesseract


# In[23]:


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# In[24]:


DATASET_PATH = r"C:\Users\DELL\OneDrive\Desktop\bajaj\lab_reports_samples"

# Enhance image for better OCR
def preprocess_image(image: Image.Image) -> Image.Image:
    image = image.convert('L')
    image = ImageEnhance.Contrast(image).enhance(2.0)
    image = ImageEnhance.Sharpness(image).enhance(2.0)
    return image


# In[25]:


from PIL import Image, ImageEnhance

def preprocess_image(image: Image.Image) -> Image.Image:
    image = image.convert('L')  # Convert to grayscale
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)  # Increase contrast
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.0)  # Increase sharpness
    return image


# In[26]:


import pytesseract

def extract_text_from_image(image: Image.Image) -> str:
    config = r'--oem 3 --psm 6'
    return pytesseract.image_to_string(image, config=config)


# In[27]:


import re

def check_out_of_range(value: str, ref_range: str) -> bool:
    try:
        numbers = re.findall(r'[\d.]+', ref_range)
        if len(numbers) == 2:
            lower, upper = float(numbers[0]), float(numbers[1])
            val = float(value)
            return val < lower or val > upper
    except:
        pass
    return False


# In[49]:


def parse_lab_test_results(text: str):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    results = []

    pattern = re.compile(
        r'^(.*?)\s+([\d.]+)\s+([^\d]*\d+[^\d]*\s*-\s*[^\d]*\d+[^\d]*)$'
    )

    for line in lines:
        match = pattern.search(line)
        if match:
            test_name = match.group(1).strip()
            value = match.group(2).strip()
            ref_range = match.group(3).strip()
            out_of_range = check_out_of_range(value, ref_range)

            results.append({
                "lab_test_name": test_name,
                "observed_value": value,
                "bio_reference_range": ref_range,
                "lab_test_out_of_range": out_of_range
            })

    return results


# In[37]:


import os
from PIL import Image

# Path to your input image
input_path = r"C:\Users\DELL\OneDrive\Desktop\bajaj\lab_reports_samples\lbmaske\AHD-0425-PA-0008061_E-mahendrasinghdischargecard_250427_1114@E.pdf_page_27.png"
# Path to save the enhanced output
output_path = r"C:\Users\DELL\OneDrive\Desktop\bajaj\lab_reports_samples\lbmaske\enhanced_output.png"

try:
    with Image.open(input_path) as img:
        print(f"Enhancing image: {os.path.basename(input_path)}")
        enhanced_img = preprocess_image(img)

        # Save the enhanced image
        enhanced_img.save(output_path)
        print(f"Enhanced image saved to: {output_path}")

        # Optionally show the enhanced image
        enhanced_img.show()

except Exception as e:
    print(f"Error enhancing image: {e}")


# In[50]:


import os
from PIL import Image

# Full path to your test image
file_path = r"C:\Users\DELL\OneDrive\Desktop\bajaj\lab_reports_samples\lbmaske\AHD-0425-PA-0008061_E-mahendrasinghdischargecard_250427_1114@E.pdf_page_27.png"

try:
    with Image.open(file_path) as img:
        print(f"\nProcessing: {os.path.basename(file_path)}")
        img = preprocess_image(img)
        text = extract_text_from_image(img)
        results = parse_lab_test_results(text)

        if results:
            for r in results:
                print(f"Test Name: {r['lab_test_name']}")
                print(f"Observed Value: {r['observed_value']}")
                print(f"Reference Range: {r['bio_reference_range']}")
                print(f"Out of Range: {r['lab_test_out_of_range']}")
                print("---")
        else:
            print("No lab test data found.")
except Exception as e:
    print(f"Error processing {file_path}: {e}")


# In[51]:


import os
from PIL import Image
import pandas as pd

# Full path to your test image
file_path = r"C:\Users\DELL\OneDrive\Desktop\bajaj\lab_reports_samples\lbmaske\AHD-0425-PA-0008061_E-mahendrasinghdischargecard_250427_1114@E.pdf_page_27.png"

try:
    with Image.open(file_path) as img:
        print(f"\nProcessing: {os.path.basename(file_path)}")
        img = preprocess_image(img)
        text = extract_text_from_image(img)
        results = parse_lab_test_results(text)

        if results:
            df = pd.DataFrame(results)
            print("\nStructured Output:\n")
            print(df)
        else:
            print("No lab test data found.")
except Exception as e:
    print(f"Error processing {file_path}: {e}")


# In[47]:


import os
import re
from PIL import Image, ImageEnhance
import pytesseract
import numpy as np
import cv2
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from base64 import b64encode
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import uvicorn

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Dataset path
DATASET_PATH = r"C:\Users\DELL\OneDrive\Desktop\bajaj\lab_reports_samples"

def preprocess_image(pil_image):
    img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(thresh)

def extract_text_from_image(pil_image):
    return pytesseract.image_to_string(pil_image)

def check_out_of_range(value, ref_range):
    try:
        value = float(value)
        numbers = [float(n) for n in re.findall(r"[-+]?\d*\.\d+|\d+", ref_range)]
        if len(numbers) == 2:
            return value < numbers[0] or value > numbers[1]
    except:
        return False
    return False

def parse_lab_test_results(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    results = []
    pattern = re.compile(r'^(.*?)([:\-]?)\s*([\d.]+)\s*(\[\w*\])?\s*([a-zA-Z/%μulUL\d]+)?\s*([<>=]*\s*[\d.]+[-–]\s*[\d.]+)?')

    for line in lines:
        match = pattern.match(line)
        if match:
            test_name = match.group(1).strip(" :.-")
            test_value = match.group(3)
            unit = match.group(5) or ""
            ref_range = match.group(6) or ""
            out_of_range = check_out_of_range(test_value, ref_range)

            results.append({
                "test_name": test_name,
                "test_value": test_value,
                "bio_reference_range": ref_range,
                "test_unit": unit,
                "lab_test_out_of_range": out_of_range
            })

    return {
        "is_success": True,
        "data": results
    }

# FastAPI app
app = FastAPI()

# Create static directory if it doesn't exist
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Keep only one root endpoint that serves the HTML
@app.get("/", response_class=HTMLResponse)
async def root():
    with open("templates/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/get-lab-tests")
async def get_lab_tests(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Process and enhance image
        processed_image = preprocess_image(image)
        
        # Convert processed image to base64 for display
        buffered = io.BytesIO()
        processed_image.save(buffered, format="PNG")
        img_str = b64encode(buffered.getvalue()).decode()
        
        # Extract text and parse results
        text = extract_text_from_image(processed_image)
        results = parse_lab_test_results(text)
        
        # Add the enhanced image to the response
        results["enhanced_image"] = f"data:image/png;base64,{img_str}"
        
        return JSONResponse(content=results)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    # Process a single test image
    file_path = r"C:\Users\DELL\OneDrive\Desktop\bajaj\lab_reports_samples\lbmaske\AHD-0425-PA-0008061_E-mahendrasinghdischargecard_250427_1114@E.pdf_page_27.png"
    
    try:
        with Image.open(file_path) as img:
            print(f"\nProcessing: {os.path.basename(file_path)}")
            img = preprocess_image(img)
            text = extract_text_from_image(img)
            results = parse_lab_test_results(text)

            if results and results.get("data"):
                df = pd.DataFrame(results["data"])
                print("\nStructured Output:\n")
                print(df)
            else:
                print("No lab test data found.")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

    # Start the FastAPI server
    print("\nStarting FastAPI server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)




