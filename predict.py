import joblib
import pandas as pd
# from le_kingmakers 

from google.cloud import storage


PATH_TO_LOCAL_MODEL = 'le_kingmakers/model2.joblib'

# AWS_BUCKET_TEST_PATH = "s3://wagon-public-datasets/taxi-fare-test.csv"

# path = "gs://{}/{}".format(BUCKET_NAME, BUCKET_TRAIN_DATA_PATH)

BUCKET_NAME = "model_lkm"  


def get_test_data(nrows, bucket=BUCKET_NAME):
    """method to get the test data (or a portion of it) from google cloud bucket
    To predict we can either obtain predictions from train data or from test data"""
    client = storage.Client()
    bucket = client.get_bucket(bucket)
    storage_location = '2021_master.csv'
    blob=bucket.blob(storage_location)
    blob.download_to_filename(storage_location)
    
    # path = "gs://le-kingmakers/2021_master.csv"
    # path = "gs://{}/".format(BUCKET_NAME,)
    # blob=client.blob('2021_master.csv')
    # ⚠️ to test from actual KAGGLE test set for submission
    df = pd.read_csv(storage_location, nrows=nrows)
    return df

# def get_data_from_gcp2(nrows=10000, local=False, optimize=False, **kwargs):
#     """method to get the training data (or a portion of it) from google cloud bucket"""
#     # Add Client() here
#     client = storage.Client()
#     if local:
#         path = LOCAL_PATH
#     else:
#         path = "gs://wagon-bootcamp-280708/train_1k.csv"
#     df = pd.read_csv(path, nrows=nrows)
#     return df


# def download_model(model_directory="PipelineTest", bucket=BUCKET_NAME, rm=True):
#     client = storage.Client().bucket(bucket)

#     storage_location = '/model2.joblib'
#     blob = client.blob(storage_location)
#     blob.download_to_filename('model2.joblib')
#     print("=> pipeline downloaded from storage")
#     model = joblib.load('model2.joblib')
#     return model


def get_model(path_to_joblib):
    pipeline = joblib.load(path_to_joblib)
    return pipeline

if __name__ == '__main__':
    x = get_test_data(100)
    print(x.head(5))
