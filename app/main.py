import os
import requests
from typing import List, Union
from pydantic import BaseModel, HttpUrl, validator
from fastapi import FastAPI, UploadFile, HTTPException, Form, Body
from fastapi.responses import JSONResponse
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import io
from dotenv import load_dotenv
from enum import Enum

load_dotenv()

app = FastAPI()

# Load environment variables
MODEL_NAME = os.getenv("MODEL_NAME", "openai/clip-vit-base-patch32")
DEVICE = os.getenv("DEVICE", "cpu")

# Load the model and processor
model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

class ImageURL(BaseModel):
    url: HttpUrl
    id: int = None

class ImageSourceType(Enum):
    LOCAL = "local"
    UPLOADED = "uploaded"
    REMOTE = "remote"

@app.post("/embed")
async def embed_image(
    files: list[UploadFile] = None,
    urls: List[str] = Body(None)
):
    print(f"Received files: {files}")
    print(f"Received URLs: {urls}")
    if not files and not urls:
        raise HTTPException(status_code=400, detail="No files or URLs provided test")
    
    results = []
    
    # Process uploaded files
    if files:
        for file in files:
            try:
                image = await get_image_from_source(file, ImageSourceType.UPLOADED)
                embedding = process_image(image)
                
                results.append({
                    "path": file.filename,
                    "embedding": embedding,
                    "id": None
                })
            except HTTPException as e:
                raise HTTPException(status_code=e.status_code, detail=f"{file.filename}: {e.detail}")
    
    # Process URLs
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
                    "embedding": embedding,
                    "id": None
                })
            except HTTPException as e:
                raise HTTPException(status_code=e.status_code, detail=f"{url}: {e.detail}")
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid URL format: {url}. {str(e)}")
    
    # Generate SQL query
    sql_query = generate_sql_query(results)
    
    # If only one file was provided, maintain backward compatibility
    if len(results) == 1:
        return JSONResponse(content={
            "embedding": results[0]["embedding"],
            "sql_query": sql_query
        })
    
    return JSONResponse(content={
        "embeddings": [{"path": r["path"], "embedding": r["embedding"]} for r in results],
        "sql_query": sql_query
    })

@app.post("/search")
async def search_image(
    file: UploadFile = None,
    url: str = Form(None)
):
    """
    Process a single image from either an uploaded file or URL and return its embedding vector.
    This endpoint is optimized for search operations.
    """
    if not file and not url:
        raise HTTPException(status_code=400, detail="Either file or URL must be provided")
    
    if file and url:
        raise HTTPException(status_code=400, detail="Provide either a file or URL, not both")
    
    try:
        if file:
            image = await get_image_from_source(file, ImageSourceType.UPLOADED)
            source_name = file.filename
        else:  # url is provided
            image = await get_image_from_source(url, ImageSourceType.REMOTE)
            source_name = url
        
        embedding = process_image(image)
        
        return JSONResponse(content={
            "source": source_name,
            "embedding": embedding
        })
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

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
    
    query = "INSERT INTO images (path, embedding) VALUES\n"
    
    values = []
    for r in results:
        # Format the embedding as a string representation of an array
        embedding_str = str(r["embedding"]).replace('[', 'ARRAY[').replace(']', ']')
        
        values.append(f"  ('{r['path']}', {embedding_str})")
    
    query += ",\n".join(values)
    
    return query
