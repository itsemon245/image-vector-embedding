import os
import requests
from typing import List, Union
from pydantic import BaseModel, HttpUrl, validator
from fastapi import FastAPI, UploadFile, HTTPException, Form, Body, Request, Depends
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import io
from dotenv import load_dotenv
from enum import Enum

# Import the middleware
from middleware.auth import AuthMiddleware

load_dotenv()

app = FastAPI()

# Load environment variables
MODEL_NAME = os.getenv("MODEL_NAME", "openai/clip-vit-base-patch32")
DEVICE = os.getenv("DEVICE", "cpu")
APP_KEY = os.getenv("APP_KEY")

@app.on_event("startup")
def check_env():
    if not APP_KEY:
        raise RuntimeError("APP_KEY environment variable must be set")

# Load the model and processor
from transformers import CLIPProcessor, CLIPModel
model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

# Add the middleware
app.add_middleware(AuthMiddleware)

class ImageURL(BaseModel):
    url: HttpUrl
    id: int = None

class ImageSourceType(Enum):
    LOCAL = "local"
    UPLOADED = "uploaded"
    REMOTE = "remote"

@app.post("/embed")
async def embed_image(
    urls: List[str] = None
):
    print(f"Received URLs: {urls}")
    
    if not urls:
        raise HTTPException(status_code=400, detail="No URLs provided")
    
    results = []
    
    if urls:
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
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    
    # Normalize the features - results in a 512-dimensional vector for clip-vit-base-patch32
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    return image_features[0].tolist()

def generate_sql_query(results):
    """Generate an SQL INSERT query for the embeddings."""
    if not results:
        return ""
    
    query = "INSERT INTO images (path, embedding) VALUES "
    
    values = []
    for r in results:
        # Convert the embedding to a string representation
        embedding_str = str(r["embedding"])
        
        # Escape single quotes in the path
        path = r['path'].replace("'", "''")
        
        values.append(f"  ('{path}', '{embedding_str}')")
    
    query += ", ".join(values)
    
    return query


# @app.post("/search")
# async def search_image(
#     url: str = Form(...)
# ):
#     """
#     Process a single image from a URL and return its embedding vector as a string.
#     This endpoint is optimized for search operations.
#     """
#     if not url:
#         raise HTTPException(status_code=400, detail="URL must be provided")
#     
#     try:
#         image = await get_image_from_source(url, ImageSourceType.REMOTE)
#         embedding = process_image(image)
#         
#         # Convert embedding to string
#         embedding_str = str(embedding)
#         
#         return JSONResponse(content={
#             "source": url,
#             "embedding": embedding_str
#         })
#     except HTTPException as e:
#         raise e
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
