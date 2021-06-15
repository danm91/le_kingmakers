import joblib
import pandas as pd
# from le_kingmakers 

from google.cloud import storage


PATH_TO_LOCAL_MODEL = 'le_kingmakers/model2.joblib'

BUCKET_NAME = "model_lkm"  

def get_test_data(nrows, bucket=BUCKET_NAME):
    """method to get the test data (or a portion of it) from google cloud bucket
    To predict we can either obtain predictions from train data or from test data"""
    client = storage.Client()
    bucket = client.get_bucket(bucket)
    storage_location = '2021_master.csv'
    blob=bucket.blob(storage_location)
    blob.download_to_filename(storage_location)
    
    df = pd.read_csv(storage_location, nrows=nrows)
    return df


def get_model(bucket=BUCKET_NAME):
    client = storage.Client()
    bucket = client.get_bucket(bucket)
    storage_location = 'model2.joblib'
    blob=bucket.blob(storage_location)
    blob.download_to_filename('NB_model.joblib')
    model = joblib.load('NB_model.joblib')
    return model


if __name__ == '__main__':
    x = get_test_data(100)
    model = get_model()
    x['prediction'] = model.predict(x['tweet'])
    print(x.head())
