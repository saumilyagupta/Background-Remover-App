from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import shutil
import os
from scripts.predict import Predictor  # Import your UNet Predictor class
import uvicorn

app = FastAPI()

# Set up static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
templates = Jinja2Templates(directory="templates")

# Create uploads directory if not exists
os.makedirs("uploads", exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/")
async def upload_images(
    source: UploadFile = File(...),
    background: UploadFile = None,  # Make background optional
    background_url: str = Form(None)  # Allow default background selection via URL
):
    source_path = f"uploads/{source.filename}"
    
    # Save the source image
    with open(source_path, "wb") as buffer:
        shutil.copyfileobj(source.file, buffer)

    # Determine background source
    if background:  # Use uploaded background
        background_path = f"uploads/{background.filename}"
        with open(background_path, "wb") as buffer:
            shutil.copyfileobj(background.file, buffer)
    elif background_url:  # Use default background from a provided URL
        background_path = f"{background_url}"  # Ensure this URL/path is accessible
    else:
        return {"error": "No background provided!"}, 422

    # Load images
    source_img = Image.open(source_path).convert("RGB")
    background_img = Image.open(background_path).convert("RGB")

    # Process image using your UNet model
    predictor = Predictor("unet.pth")
    processed_img = predictor.background_changed_img(source_img, background_img)
    
    # Convert tensor to PIL Image and save
    output_path = f"uploads/processed_{source.filename}"
    output_pil = Image.fromarray((processed_img.squeeze().cpu().numpy().transpose(1, 2, 0) * 255).astype('uint8'))
    output_pil.save(output_path)

    return {"message": "Images uploaded successfully!", "output_path": output_path}

@app.post("/upload_video/")
async def upload_video(
    video: UploadFile = File(...),
    background: UploadFile = None,  # Optional uploaded background for video
    background_url: str = Form(None)  # Or default background URL
):
    video_path = f"uploads/{video.filename}"
    
    # Save the video file
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    # Determine background for video processing
    if background:
        background_path = f"{background.filename}"
        with open(background_path, "wb") as buffer:
            shutil.copyfileobj(background.file, buffer)
    elif background_url:
        background_path = f"{background_url}"  # e.g., a path like '/static/backgrounds/bg1.jpg'
    else:
        return {"error": "No background provided!"}, 422

    # Load the background image as PIL Image
    background_img = Image.open(background_path).convert("RGB")

    # Process the video using your Predictor's video background change method
    predictor = Predictor("unet.pth")
    output_video_path = f"uploads/processed_{video.filename}"
    output_video = predictor.background_chnage_video(video_path, background_img, output_video_path)

    return {"message": "Video processed successfully!", "output_path": output_video}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=2221, reload=True)
