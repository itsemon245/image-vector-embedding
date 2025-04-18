import os
import requests
from typing import List
from PIL import Image
import torch
import io
from dotenv import load_dotenv
from enum import Enum

load_dotenv()
# Load environment variables
MODEL_NAME = os.getenv("MODEL_NAME", "openai/clip-vit-base-patch32")
DEVICE = os.getenv("DEVICE", "cpu")
APP_KEY = os.getenv("APP_KEY")
PORT = int(os.getenv("PORT", 8787))
HOST = os.getenv("HOST", "0.0.0.0")

#add a colors dictionary
colors = {
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "magenta": "\033[95m",
    "cyan": "\033[96m",
    "white": "\033[97m",
    "reset": "\033[0m"
}

# add a function to print with colors
def cPrint(color, message):
    print(f"{colors[color]}{message}{colors['reset']}")



# Check for env variables
if not APP_KEY:
    cPrint("red", "APP_KEY environment variable must be set")
    raise RuntimeError("APP_KEY environment variable must be set")

# Load the model and processor
# add colors to the print statements
cPrint("green", f"Loading model {MODEL_NAME} on device {DEVICE}...")
from transformers import CLIPProcessor, CLIPModel
model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)


from pydantic import BaseModel, HttpUrl, validator
from fastapi import FastAPI, UploadFile, HTTPException, Form, Body, Request, Depends
from fastapi.responses import JSONResponse

app = FastAPI()

# Import the middleware
from middleware.auth import AuthMiddleware
# Add the middleware
app.add_middleware(AuthMiddleware)

class ImageSourceType(Enum):
    LOCAL = "local"
    UPLOADED = "uploaded"
    REMOTE = "remote"

@app.post("/embed")
async def embed_image(
    urls: List[str] = None
):
    cPrint("yellow", f"Received URLs: {urls}")
    
    if not urls:
        raise HTTPException(status_code=400, detail="No URLs provided")
    
    results = []
    
    for url in urls:
        try:
            # Validate URL format
            if not url.startswith(('http://', 'https://')):
                raise ValueError("URL must start with http:// or https://")
            
            image = await get_image_from_source(url, ImageSourceType.REMOTE)
            embedding = process_image(image)
            
            results.append({
                "path": url,
                "embedding": str(embedding),  # Convert embedding to string
                "id": None
            })
        except HTTPException as e:
            raise HTTPException(status_code=e.status_code, detail=f"{url}: {e.detail}")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid URL format: {url}. {str(e)}")
    
    # If only one file was provided, maintain backward compatibility
    if len(results) == 1:
        return JSONResponse(content={
            "embedding": results[0]["embedding"]
        })
    
    return JSONResponse(content={
        "embeddings": [{"path": r["path"], "embedding": r["embedding"]} for r in results]
    })


async def get_image_from_source(source, source_type: ImageSourceType):
    """
    Get an image from various sources: uploaded file, local path, or remote URL.
    
    Args:
        source: Either an UploadFile, a file path string, or a URL string
        source_type: Enum specifying the type of source (LOCAL, UPLOADED, or REMOTE)
    
    Returns:
        PIL.Image: The loaded image
    """
    try:
        if source_type == ImageSourceType.UPLOADED:  # UploadFile object
            image_bytes = await source.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        elif source_type == ImageSourceType.REMOTE:  # Remote URL
            response = requests.get(str(source), timeout=10)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
        elif source_type == ImageSourceType.LOCAL:  # Local file path
            image = Image.open(source).convert("RGB")
        else:
            raise ValueError(f"Invalid source type: {source_type}")
        
        return image
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

def process_image(image):
    """Process an image and return its embedding vector."""
    try:
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        
        # Normalize the features
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features[0].tolist()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    config = uvicorn.Config("main:app", port=PORT, log_level="info")
    server = uvicorn.Server(config)
    server.run()
