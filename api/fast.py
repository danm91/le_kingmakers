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
import twint
import random
from scipy.special import softmax

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
    if os.path.exists("roberta.joblib"):
        return joblib.load('roberta.joblib')
    client = storage.Client()
    bucket = client.get_bucket(bucket)
    storage_location = 'roberta_model.joblib'
    blob=bucket.blob(storage_location)
    blob.download_to_filename('roberta.joblib')
    model = joblib.load('roberta.joblib')
    return model

def get_tokenizer(bucket=BUCKET_NAME):
    if os.path.exists('tokenizer.joblib'):
        return joblib.load('tokenizer.joblib')
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

def roberta_encode(data, maximum_length) :
    input_ids = []
    attention_masks = []

    for text in data:
        encoded = tokenizer.encode_plus(
            text, 
            add_special_tokens=True,
            max_length=maximum_length,
            pad_to_max_length=True,

            return_attention_mask=True,
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
        
    return np.array(input_ids),np.array(attention_masks)

def twint_scrape(search,output_file, n):
    
    c = twint.Config()

    c.Search = search
    c.Store_csv = True
    c.Lang = 'en'
    c.Limit = n
    c.Output = output_file
    c.Count = True
    c.Hide_output = True

    return twint.run.Search(c)

def shorten_time(input):
    return input[:16]

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

@app.get("/scrapeandpredict")
def scrape_n_predict(search: str, count: int):
    output_file = search + str(random.randint(0, 9999999)) + '.csv'
    twint_scrape(search,output_file, n = count)
    df = pd.read_csv(output_file)
    df['clean'] = df['tweet'].apply(preprocess)
    
    #find maxlen for encoding
    
    X_train_token = [i.split() for i in df['clean']]
    maxlen = len(X_train_token[0])
    for i in X_train_token:
        if len(i) > maxlen:
            maxlen = len(i) 
    
    #encode tweets for model
    
    input_ids, attention_masks = roberta_encode(df['clean'],maxlen)
    
    #predict log odds negative, neutral and positive sentiment
    
    predictions = model.predict([input_ids, attention_masks], batch_size=110, verbose=1, use_multiprocessing=True, workers=8)
    
    #convert log odds to probabilities 
    #score = prob(negative) - prob(positive) 
    scores = []
    for i in predictions[0]:
        softmax_val = softmax(i)
        scores.append(softmax_val[2] - softmax_val[0])
    
    #append scores to the dataframe
    df['scores'] = scores
    
    #create count column 
    #count = 1 + number of retweets + 0.5(number of likes)
    #a retweet is equivalent to a tweet. A like equivalent to half a tweet. 
    #df['count'] = 1 + df['retweets_count'].astype('int64') + 0.5*df['likes_count'].astype('int64')
    
    #multiply score by respective count, will standardize later. 
    #df['scores'] = df['scores'] * df['count']
    #ignore seconds 
    #df['created_at'] = df['created_at'].apply(shorten_time)
    
    #group tweets by time of creation
    #grouped = df[['scores', 'created_at', 'count']].groupby(['created_at']).sum().reset_index()
    #get the mean scores
    #grouped['scores'] = grouped['scores'] / grouped['count']
    #convert 'created_at' to datetime datatype
    #grouped['created_at'] = pd.to_datetime(grouped['created_at'], format='%Y-%m-%d %H:%M')
    #return as json file
    return df.to_json()