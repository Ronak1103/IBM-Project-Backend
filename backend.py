from fastapi import FastAPI, File, UploadFile
import uvicorn
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf  

app = FastAPI()

# Load the pre-trained deep learning model
MODEL_PATH = "model/64-4.h5"  
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "pituitary"]

# Image preprocessing function
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# API endpoint for image upload and prediction
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("L")
        # Preprocess image
        processed_image = preprocess_image(image)

        # Perform prediction
        predictions = model.predict(processed_image)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]  # Get highest probability class
        confidence = float(np.max(predictions))  # Get confidence score

        return {
            "filename": file.filename,
            "predicted_class": predicted_class,
            "confidence": confidence
        }
    
    except Exception as e:
        return {"error": str(e)}