import logging
import os
import io
import json
import tempfile
import time
from dotenv import load_dotenv
from PIL import Image, ImageEnhance
from google.cloud import vision
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pdf2image import convert_from_bytes
import httpx
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3-70b-8192")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")

# Setup logger
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

# FastAPI app setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[BACKEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool executor for blocking OCR calls
executor = ThreadPoolExecutor(max_workers=2)


def setup_google_credentials():
    raw_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    if not raw_creds:
        logging.error("GOOGLE_APPLICATION_CREDENTIALS not set.")
        raise HTTPException(status_code=500, detail="Google credentials not configured.")

    if os.path.exists(raw_creds):
        logging.info("Using Google credentials from file path.")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = raw_creds
        return

    try:
        creds_data = json.loads(raw_creds)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp:
            json.dump(creds_data, temp)
            temp.flush()
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp.name
        logging.info("Google credentials loaded from inline JSON and written to temp file.")
    except Exception:
        logging.exception("Failed to parse GOOGLE_APPLICATION_CREDENTIALS as JSON.")
        raise HTTPException(status_code=500, detail="Invalid Google credentials format.")


def preprocess_image(img_bytes: bytes) -> bytes:
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("L")
        base_width = 1200
        w_percent = base_width / float(img.size[0])
        h_size = int(float(img.size[1]) * w_percent)
        img = img.resize((base_width, h_size), Image.LANCZOS)
        img = ImageEnhance.Contrast(img).enhance(2.5)
        output = io.BytesIO()
        img.save(output, format="PNG")
        return output.getvalue()
    except Exception:
        logging.warning("Preprocessing failed, using original image.")
        return img_bytes

async def groq_spellcheck(raw_text: str) -> str:
    prompt = f"""
You are a spelling correction assistant.

Correct only the spelling mistakes in the text below. Do not change any grammar, punctuation, formatting, or phrasing.

Return only the corrected text as plain output — no explanations, no headings, no markdown, and no prefix like "Here is...".

Text:
\"\"\"{raw_text}\"\"\"
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 2000,
    }

    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                res = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload
                )
                if res.status_code == 200:
                    result = res.json()["choices"][0]["message"]["content"].strip()

                    # Optional post-processing: strip unwanted intro
                    lower = result.lower()
                    if lower.startswith("here is") or lower.startswith("corrected text"):
                        parts = result.split("\n", 1)
                        if len(parts) > 1:
                            result = parts[1].strip()

                    logging.info(f"Groq API success. Corrected text (truncated): {result[:250]}...")
                    return result

                logging.warning(f"Groq API failed (status {res.status_code}): {res.text}")
        except httpx.RequestError as e:
            logging.warning(f"Groq API request error (attempt {attempt + 1}): {e}")
        await asyncio.sleep(1.5)

    raise HTTPException(status_code=500, detail="Groq spell correction failed.")


def vision_task(content: bytes) -> str:
    setup_google_credentials()
    logging.info("Calling Google Vision API...")
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)
    if response.error.message:
        logging.error(f"Vision API error: {response.error.message}")
        raise HTTPException(status_code=500, detail="Vision API failed.")
    texts = response.text_annotations
    extracted = texts[0].description if texts else ""
    logging.info(f"OCR Result (truncated): {extracted[:250]}...")
    return extracted


async def extract_text_from_image_or_pdf(file: UploadFile) -> str:
    filename = file.filename.lower()
    content = await file.read()

    image_bytes_list = []
    if filename.endswith(".pdf"):
        logging.info("File is a PDF. Converting pages to images...")
        try:
            images = convert_from_bytes(content, fmt="png")
            for img in images:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                image_bytes_list.append(buf.getvalue())
        except Exception as e:
            logging.error(f"PDF conversion failed: {e}")
            raise HTTPException(status_code=400, detail="Invalid or unreadable PDF file.")
    else:
        logging.info("File is an image.")
        image_bytes_list = [content]

    full_raw_text = ""
    for idx, img_bytes in enumerate(image_bytes_list):
        processed = preprocess_image(img_bytes)
        try:
            ocr_text = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(executor, vision_task, processed),
                timeout=20.0,
            )
            logging.info(f"OCR text from image {idx + 1} received.")
            full_raw_text += ocr_text + "\n"
        except asyncio.TimeoutError:
            logging.warning(f"OCR timeout on image {idx + 1}. Skipping.")

    if not full_raw_text.strip():
        raise HTTPException(status_code=422, detail="No text found in input.")

    corrected_text = await groq_spellcheck(full_raw_text)
    return corrected_text


@app.post("/getTextFromImage/")
async def extract_text(file: UploadFile = File(...)):
    try:
        logging.info(f"Received file: {file.filename} ({file.content_type})")
        start_time = time.time()
        corrected_text = await extract_text_from_image_or_pdf(file)
        elapsed = time.time() - start_time
        logging.info(f"Text extraction and correction completed in {elapsed:.2f}s")
        return {"extracted_text": corrected_text}
    except HTTPException as e:
        logging.warning(f"Request failed with {e.status_code}: {e.detail}")
        raise e
    except Exception:
        logging.exception("Unexpected error during /getTextFromImage/")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/")
async def root():
    return {"message": "Text Extractor Service is running"}
