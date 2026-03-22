from fastapi import FastAPI
import logging
import boto3
from botocore.exceptions import ClientError

with open("../keys/s3-key-id", "r") as f:
    access_key = f.readline().rstrip()

with open("../keys/s3-key", "r") as f:
    secret_key = f.readline().rstrip()


bucket_name = "full-images"

def create_presigned_post(
    object_name,
    bucket_name = bucket_name,
    fields=None,
    conditions=None,
    expiration=600
):
    """Generate a presigned URL S3 POST request to upload a file

    :param bucket_name: string
    :param object_name: string
    :param fields: Dictionary of prefilled form fields
    :param conditions: List of conditions to include in the policy
    :param expiration: Time in seconds for the presigned URL to remain valid
    :return: Dictionary with the following keys:
        url: URL to post to
        fields: Dictionary of form fields and values to submit with the POST
    :return: None if error.
    """

    # Generate a presigned S3 POST URL
    s3_client = boto3.client(service_name='s3',
                             endpoint_url='https://storage.yandexcloud.net',
                             aws_access_key_id=access_key,
                             aws_secret_access_key=secret_key)
    try:
        response = s3_client.generate_presigned_post(
            bucket_name,
            object_name,
            Fields=fields,
            Conditions=conditions,
            ExpiresIn=expiration,
        )
    except ClientError as e:
        logging.error(e)
        return None

    # The response contains the presigned URL and required fields
    return response



app = FastAPI()


@app.get("/image_url/{image_path}")
async def basic(image_path: str):
    response = create_presigned_post(image_path)
    return response
