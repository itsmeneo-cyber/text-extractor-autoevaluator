import logging
import os
import io
import json
import tempfile
from dotenv import load_dotenv
from PIL import Image, ImageEnhance
from google.cloud import vision
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pdf2image import convert_from_bytes
import httpx
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Load env variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3-70b-8192")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")

# Setup FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[BACKEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logger setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Thread pool executor
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

    # Handle inline JSON format
    try:
        creds_data = json.loads(raw_creds)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp:
            json.dump(creds_data, temp)
            temp.flush()
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp.name
        logging.info("Google credentials loaded from inline JSON and written to temp file.")
    except Exception as e:
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

Please only correct spelling mistakes in the following text. Do NOT change any grammar, phrasing, or meaning. Return the corrected text only.

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
                res = await client.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
                if res.status_code == 200:
                    return res.json()["choices"][0]["message"]["content"].strip()
                logging.warning(f"Groq API failed (status {res.status_code}): {res.text}")
        except httpx.RequestError as e:
            logging.warning(f"Groq API request error (attempt {attempt + 1}): {e}")
        await asyncio.sleep(1.5)

    raise HTTPException(status_code=500, detail="Groq spell correction failed.")


def vision_task(content: bytes) -> str:
    setup_google_credentials()
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)
    if response.error.message:
        logging.error(f"Vision API error: {response.error.message}")
        raise HTTPException(status_code=500, detail="Vision API failed.")
    texts = response.text_annotations
    return texts[0].description if texts else ""


async def extract_text_from_image_or_pdf(file: UploadFile) -> str:
    filename = file.filename.lower()
    content = await file.read()

    image_bytes_list = []
    if filename.endswith(".pdf"):
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
        image_bytes_list = [content]

    full_raw_text = ""
    for img_bytes in image_bytes_list:
        processed = preprocess_image(img_bytes)
        try:
            ocr_text = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(executor, vision_task, processed),
                timeout=20.0,
            )
            full_raw_text += ocr_text + "\n"
        except asyncio.TimeoutError:
            logging.warning("OCR timeout on one image.")
            continue

    if not full_raw_text.strip():
        raise HTTPException(status_code=422, detail="No text found in input.")

    corrected_text = await groq_spellcheck(full_raw_text)
    return corrected_text


@app.post("/getTextFromImage/")
async def extract_text(file: UploadFile = File(...)):
    try:
        corrected_text = await extract_text_from_image_or_pdf(file)
        return {"extracted_text": corrected_text}
    except HTTPException as e:
        raise e
    except Exception:
        logging.exception("Unexpected error.")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
async def health_check():
    return {"status": "ok"}
