import os
from google.cloud import storage

def upload_blob(source_file_name, 
                folder_name = None, 
                bucket_name='ds-ml-nlp', 
                run_locally=False):
    """
    Uploads a file to Google Cloud Storage
    """
    
    storage_client = storage.Client()

    bucket = storage_client.get_bucket(bucket_name)
    ### If user specifies a specific folder, append that
    if folder_name:
        blob_location = f'{folder_name}/{source_file_name}'
    else:
        blob_location = f'{source_file_name}'

    ### Upload the file
    blob = bucket.blob(blob_location)
    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {blob_location}.")

def download_blob(source_blob_name, 
                    destination_file_name, 
                    bucket_name='ds-ml-nlp', 
                    run_locally=False, 
                    folder=False):
    """
    Downloads a file from Google Cloud Storage
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Blob {blob.name} downloaded to {destination_file_name}.")