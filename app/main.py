import os
import requests
from typing import List
from PIL import Image
import torch
import io
from dotenv import load_dotenv
from enum import Enum
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware  # Add CORS support

# Load environment variables first
load_dotenv()
MODEL_NAME = os.getenv("MODEL_NAME", "openai/clip-vit-base-patch32")
DEVICE = os.getenv("DEVICE", "cpu")
APP_KEY = os.getenv("APP_KEY")
PORT = int(os.getenv("PORT", 8787))
HOST = os.getenv("HOST", "0.0.0.0")

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and add the auth middleware
from middleware.auth import AuthMiddleware
app.add_middleware(AuthMiddleware)

# Console colors
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

def cPrint(color, message):
    print(f"{colors[color]}{message}{colors['reset']}")

# Validate environment variables
if not APP_KEY:
    cPrint("red", "APP_KEY environment variable must be set")
    raise RuntimeError("APP_KEY environment variable must be set")

# Load ML models
cPrint("green", f"Loading model {MODEL_NAME} on device {DEVICE}...")
from transformers import CLIPProcessor, CLIPModel
model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

class ImageSourceType(Enum):
    LOCAL = "local"
    UPLOADED = "uploaded"
    REMOTE = "remote"

# Helper functions
async def get_image_from_source(source, source_type: ImageSourceType):
    """Get an image from various sources."""
    try:
        if source_type == ImageSourceType.UPLOADED:
            image_bytes = await source.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        elif source_type == ImageSourceType.REMOTE:
            # Add timeout and better error handling for requests
            response = requests.get(
                str(source), 
                timeout=30,
                headers={'User-Agent': 'Mozilla/5.0'}  # Add user agent
            )
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
        elif source_type == ImageSourceType.LOCAL:
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

# Routes
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/embed")
async def embed_image(urls: List[str] = None):
    cPrint("cyan", f"Received URLs: {urls}")
    
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
                "embedding": str(embedding),
                "id": None
            })
        except HTTPException as e:
            raise HTTPException(status_code=e.status_code, detail=f"{url}: {e.detail}")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid URL format: {url}. {str(e)}")
    
    # If only one file was provided, maintain backward compatibility
    if len(results) == 1:
        return JSONResponse(content={"path": results[0]["path"], "embedding": results[0]["embedding"]})
    
    return JSONResponse(content={
        "embeddings": [{"path": r["path"], "embedding": r["embedding"]} for r in results]
    })

# Server startup
if __name__ == "__main__":
    import uvicorn
    config = uvicorn.Config(
        "main:app",
        host=HOST,
        port=PORT,
        reload=True,
        log_level="info",
        timeout_keep_alive=65,  # Increase keep-alive timeout
        workers=4  # Add multiple workers
    )
    server = uvicorn.Server(config)
    server.run()
