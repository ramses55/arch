import boto3
import ultralytics
import cv2
import os
import time
from utils import result
import json


with open("./keys/s3-key-id", "r") as f:
    access_key = f.readline().rstrip()

with open("./keys/s3-key", "r") as f:
    secret_key = f.readline().rstrip()



# Generate a presigned S3 POST URL
s3 = boto3.client(service_name='s3',
                         endpoint_url='https://storage.yandexcloud.net',
                         aws_access_key_id=access_key,
                         aws_secret_access_key=secret_key)



while (True):
    
    QUEUE_URL = "https://message-queue.api.cloud.yandex.net/b1gn2cep80rfd8mos87v/dj60000000ifp1rm0516/queue"
    
    sqs = boto3.client(
        "sqs",
        endpoint_url="https://message-queue.api.cloud.yandex.net",
        region_name="ru-central1",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )
    
    
    
    response = sqs.receive_message(
        QueueUrl=QUEUE_URL,
        MaxNumberOfMessages=10,
        VisibilityTimeout=30,
        WaitTimeSeconds=20,
        ReceiveRequestAttemptId='string'
    )
    
    if 'Messages' in response:
        #print(response['Messages'])
        print("Got a message!")
    else:
        print("No messages yet!")
        continue


    f = open("result.csv", "a")

    for message in response["Messages"]:
        body_info = json.loads(message["Body"])
        bucket_name = body_info["bucket"]
        file_name = body_info["key"]
        receipt_handle = message['ReceiptHandle']
        local_name = "/tmp/" + file_name
        s3.download_file(bucket_name, file_name, local_name)
    
    
    
        model = ultralytics.YOLO("./best.pt")
        
        time.sleep(0.5)
        print(local_name)
        output = model(local_name,
                    conf=0.1,
                    save=False,
                    show=False,
                    verbose=False)
        
        r = result(output[0])
        r.all()
        m = r.csv_res()
        im = r.draw()
        success = cv2.imwrite(f"m-{os.path.basename(file_name)}", im)
        print(success)
        print(m,file=f)
        response_del = sqs.delete_message(QueueUrl=QUEUE_URL,
                                         ReceiptHandle=receipt_handle)
        
    f.close()
