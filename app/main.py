from fastapi import FastAPI
import tensorflow as tf

app = FastAPI()

# Later, load correct model e.g. from MLflow and use that instead for predictions
model = tf.keras.models.load_model("../models/cat_classifier.keras")


@app.get("/")
async def index():
    return {"message": "index page"}


@app.post("/predict/")
async def predict(image):
    # TODO: Prepare image for prediction (input_size etc.)
    # TODO: Make prediction
    # TODO: Return a list of prediction probabilities and labels
    # TODO: Separate model could be used to predict if cat even exists in the image before breed classification
    ...
