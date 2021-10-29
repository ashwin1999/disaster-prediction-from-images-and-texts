# FastAPI Imports
from fastapi import FastAPI, UploadFile, File
import shutil

# Deep Learning Imports
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text

# Custom Module Imports
from prediction import *
from schemas import *


# Method to Load Model from '/models/' folder
def load_model(modelname):
    model = tf.keras.models.load_model(f'models/{modelname}')
    return model


# Load both the models (text and image models)
text_model = load_model('text_prediction_bert')
image_model = load_model('image_prediction_efficientnet')


# Init FastAPI
description = """
Predict disaster/mentions of disaster from Image/Text.
"""


app = FastAPI(
    title="Disaster Prediction API",
    description=description,
    version="1.0.1",
    contact={
        "name": "Ashwin Balasubramanian",
        "url": "https://ashwinbala.ml",
        "email": "acchu99@gmail.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://github.com/Ashwin1999/Disaster-Prediction---FastAPI/blob/main/LICENSE",
    },
)


# Base Route
@app.get("/")
def root():
    return {
        "Message": "This is the base route"
    }


# Route to predict one sentence
@app.post("/one_text", response_model=onePrediction)
def make_one_prediction(data: Sentence):
    prediction = text_model.predict([data.text]).tolist()[0][0]
    res = make_text_prediction(prediction)
    return {"prediction": prediction, "pred_class": res}


# Route to predict a list of sentences
@app.post("/multi_text", response_model=multiPrediction)
def make_multi_prediction(data: Sentences):
    prediction = text_model.predict(data.texts).flatten().tolist()
    res = make_text_prediction(prediction, multi=True)
    return {"prediction": prediction, "pred_class": res}


# Route to predict an image
@app.post("/image")
async def root(file: UploadFile = File(...)):
    extension = file.filename.split('.')[1]

    with open(f"buffer/temp.{extension}", "wb") as tempfile:
        shutil.copyfileobj(file.file, tempfile)

    img = cv2.imread(f'buffer/temp.{extension}')
    img = cv2.resize(img, (465, 396))
    img = np.array([img])/255

    pred = image_model.predict(img)
    pred_class = make_image_prediction(pred)

    return {"prediction": pred_class}
