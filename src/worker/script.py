import boto3
import traceback
import time
import json

from utils import settings
from utils import work

import sys

def fun():
    oauth_token = settings.oauth_token
    access_key = settings.access_key_id
    secret_key = settings.access_key
    QUEUE_URL = settings.queue_url
    worker_url = settings.worker_url
    
    sqs = boto3.client(
        "sqs",
        endpoint_url="https://message-queue.api.cloud.yandex.net",
        region_name="ru-central1",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )
    
    img_counter = 0
    res_counter = 0
    
    
    img_limit = settings.img_limit
    res_limit = settings.res_limit
    
        
    ok = open("ok.csv", "w")
    failed = open("failed.csv", "w")
    
    h = "Исходное имя файла|Тип объекта|Размер (мм)|Число объектов|Путь к файлу\n"
    
    ok.write(h)
    failed.write(h)
    
    while(True):
        response = sqs.receive_message(
                QueueUrl=QUEUE_URL,
                MaxNumberOfMessages=10,
                VisibilityTimeout=30,
                WaitTimeSeconds=10,
                ReceiveRequestAttemptId='string'
            )
    
        
        messages = response.get("Messages", [])
    
        if len(messages) == 0:
            res_counter += 1
    
        for message in messages:
            img_counter += 1
            body_info = json.loads(message["Body"])
            path = body_info["path"]
            filename = body_info["filename"]
            receipt_handle = message['ReceiptHandle']
            try:
                time.sleep(0.3)
                print(f"Started work on  {filename}")
                key=work(path, filename, ok, failed)
                print(f"work returned: {key}")
                print(f"Ended work on  {filename}")
                response_del = sqs.delete_message(QueueUrl=QUEUE_URL,
                                                 ReceiptHandle=receipt_handle)
        
            except Exception as e:
                print(f"An error occurred: {e}")
                traceback.print_exc()
    
        print("res_counter=",res_counter)
        print("img_counter=",img_counter)
        if (res_counter > res_limit or img_counter > img_limit):
            ok.close()
            failed.close()
            path = '-'
            filename = '-'
            print(f"Started work on  {filename}")
            key=work(path, filename, None, None)
            print(f"work returned: {key}")
            print(f"Ended work on  {filename}")
            work("-", "-", None, None)
            break
    
    
    print("out of loop")
    print("img_limit:", img_limit)
    print("img_counter:", img_counter)
    
    
    
    if img_counter > img_limit:
        print("Should call next instance!", flush=True)
        return 0
    else:
        print("Should finish!", flush=True)
        return 1


if __name__ == '__main__':
    res = fun()
    if res == 0:
        sys.exit(10)
    else:
        sys.exit(0)

