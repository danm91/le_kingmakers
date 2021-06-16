#------------------------------------------------------
#   Fast API 
#------------------------------------------------------
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from # import your predict and load model function
import os
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf

# model = joblib.load('NB_model.joblib')

#------------------------------------------------------
app = FastAPI()
#------------------------------------------------------
# model, tokenizer = # Load model function with a robust path
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

#------------------------------------------------------
# to use cardiffnlp model on gcp
from google.cloud import storage
BUCKET_NAME = "model_lkm"  

def get_model(bucket=BUCKET_NAME):
    client = storage.Client()
    bucket = client.get_bucket(bucket)
    storage_location = 'roberta_model.joblib'
    blob=bucket.blob(storage_location)
    blob.download_to_filename('roberta.joblib')
    model = joblib.load('roberta.joblib')
    return model

def get_tokenizer(bucket=BUCKET_NAME):
    client = storage.Client()
    bucket = client.get_bucket(bucket)
    storage_location = 'roberta_tokenizer.joblib'
    blob=bucket.blob(storage_location)
    blob.download_to_filename('tokenizer.joblib')
    model = joblib.load('tokenizer.joblib')
    return model

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

model = get_model(bucket=BUCKET_NAME)
tokenizer = get_tokenizer(bucket=BUCKET_NAME)


#------------------------------------------------------
@app.get("/")
def index():
    return {"ok": True}
#------------------------------------------------------

@app.get("/predict")
def prediction(text: str):
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='tf')
    output = model(encoded_input)
    scores = np.asarray(tf.nn.softmax(output.logits, axis=-1))[0].tolist()
    score = scores[2] - scores[0]
    return score

def prediction(text: str):
    df_text = pd.DataFrame([text], columns=['tweet'])
    prediction = model.predict(df_text['tweet'])
    return {'prediction': str(prediction[0])}