import logging
import os
from google.cloud import vision
import io
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware  # <-- Added
from dotenv import load_dotenv
from PIL import Image, ImageEnhance
import io
import tempfile
import json
# ----------------------------
# Load Environment Variables
# ----------------------------
load_dotenv()

# ----------------------------
# Logging Setup
# ----------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ocr_debug.log"),
        logging.StreamHandler()
    ]
)

# FastAPI instance
app = FastAPI()
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:8082")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image(content: bytes) -> bytes:
    try:
        img = Image.open(io.BytesIO(content)).convert('L')
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        output = io.BytesIO()
        img.save(output, format="PNG")
        return output.getvalue()
    except Exception as e:
        logging.warning("Preprocessing failed. Using original image.")
        return content
# ----------------------------
# Function to Extract Text from Image
# ----------------------------

def extract_text_from_image(image_file):
    logging.info("Starting text extraction process.")

    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    if not credentials_path:
        logging.error("Environment variable 'GOOGLE_APPLICATION_CREDENTIALS' is not set.")
        raise HTTPException(status_code=500, detail="Google credentials are not set.")

    # If file doesn't exist (Render), treat it as embedded JSON instead of file path
    if not os.path.exists(credentials_path):
        logging.info("GOOGLE_APPLICATION_CREDENTIALS looks like JSON. Trying to interpret as embedded credentials.")

        try:
            creds_data = json.loads(credentials_path)
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as temp:
                json.dump(creds_data, temp)
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp.name
                logging.info("Temp credential file created from inline JSON.")
        except Exception as e:
            logging.exception("Failed to handle GOOGLE_APPLICATION_CREDENTIALS as embedded JSON.")
            raise HTTPException(status_code=500, detail="Invalid credential format in environment variable.")
    else:
        logging.info(f"Using credential file: {credentials_path}")

    # Initialize Vision API client
    try:
        client = vision.ImageAnnotatorClient()
        logging.info("Vision API client initialized successfully.")
    except Exception as e:
        logging.exception("Failed to initialize Vision API client.")
        raise HTTPException(status_code=500, detail="Failed to initialize Vision API client.")

    # Read image content
    try:
        content = image_file.read()
        content = preprocess_image(content)
        logging.debug(f"Read {len(content)} bytes from image.")
    except Exception as e:
        logging.exception("Failed to read the image file.")
        raise HTTPException(status_code=400, detail="Failed to read the image.")

    image = vision.Image(content=content)

    # Send image to Vision API for text detection
    try:
        logging.info("Sending image to Vision API for text detection.")
        response = client.text_detection(image=image)
    except Exception as e:
        logging.exception("Vision API request failed.")
        raise HTTPException(status_code=500, detail="Vision API request failed.")

    # Handle response
    if response.error.message:
        logging.error(f"Vision API returned an error: {response.error.message}")
        raise HTTPException(status_code=500, detail="Vision API returned an error.")

    texts = response.text_annotations

    if texts:
        logging.info("Text detected successfully.")
        return texts[0].description
    else:
        logging.warning("No text was found in the image.")
        return "No text found."
#Health Check
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# ----------------------------
# FastAPI Route for OCR
# ----------------------------
@app.post("/getTextFromImage/")
async def extract_text(file: UploadFile = File(...)):
    try:
        result = extract_text_from_image(file.file)
        return {"extracted_text": result}
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.exception("Unexpected error.")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

# ----------------------------
# Run the FastAPI app
# ----------------------------
# To run this app, use the following:
# uvicorn main:app --reload
