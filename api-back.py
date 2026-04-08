from fastapi import FastAPI, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from typing import List
import pandas as pd
import MySQLdb


import logging
import boto3
from botocore.exceptions import ClientError

with open("./keys/s3-key-id", "r") as f:
    access_key = f.readline().rstrip()

with open("./keys/s3-key", "r") as f:
    secret_key = f.readline().rstrip()


with open("./keys/mysql_pass", "r") as f:
    mysql_pass = f.readline().rstrip()

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

class FileRequest(BaseModel):
    files: List[str]



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
)




data = [
    {"name": "A", "value": 10},
    {"name": "B", "value": 20},
    {"name": "C", "value": 30},
]


conn = MySQLdb.connect(
      host="rc1b-ojtel5k6c967smm2.mdb.yandexcloud.net",
      port=3306,
      db="db1",
      user="user1",
      passwd=mysql_pass,
      ssl={'ca': './.mysql/root.crt'})





@app.get("/data")
#async def get_data(
#    skip: int = Query(0, ge=0),
#    limit: int = Query(100, ge=1, le=1000),
#    sort: str = None,
#    filter_field: str = None,
#    filter_value: str = None
#):
#    # Your database query here
#    # df = pd.read_sql("SELECT * FROM your_table", conn)
#    df = pd.DataFrame({'id': [1,2,3], 'name': ['A','B','C'], 'value': [10,20,30]})
#    
#    # Apply filters & sorting
#    if filter_field and filter_value:
#        df = df[df[filter_field].astype(str).str.contains(filter_value, case=False)]
#    if sort:
#        df = df.sort_values(sort)
#    
#    total = len(df)
#    df = df.iloc[skip:skip+limit]
#    
#    return {"data": df.to_dict(orient="records"), "total": total}



def get_data():

    cur = conn.cursor()
    cur.execute("SELECT * FROM data")
    conn.commit()
    
    rows = cur.fetchall()
    columns = [col[0] for col in cur.description]
    df = pd.DataFrame(rows, columns=columns)
    return df.to_dict(orient = "records")
    #return data

#@app.get("/", response_class=HTMLResponse)
#async def serve_ui():
#    with open("templates/index.html") as f:
#        return HTMLResponse(content=f.read())
#



@app.get("/image_url/{image_path}")
async def basic(image_path: str):
    response = create_presigned_post(image_path)
    return response


@app.post("/downloadImages")
async def imageDownload(data: FileRequest):
    print(data.files)
    
