from fastapi import FastAPI, File, UploadFile
import uvicorn
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf  # Use torch if your model is in PyTorch

app = FastAPI()

# Load the pre-trained deep learning model
MODEL_PATH = "brain_tumor_model.h5"  # Change this to your actual model path
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels (Update this based on your model)
CLASS_NAMES = ["No Tumor", "Meningioma", "Glioma", "Pituitary Tumor"]

# Image preprocessing function
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))  # Resize to match model input size
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# API endpoint for image upload and prediction
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")  # Convert to RGB

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

# Run FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
