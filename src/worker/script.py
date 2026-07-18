import boto3
import traceback
import time
import json

from utils import settings
from utils import work, download_b, mv
from utils import upload_part, model_part, download_part

import sys

import threading
from concurrent.futures import ThreadPoolExecutor
from itertools import count


import queue
import cv2
import numpy as np


import ultralytics


import requests
import logging


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

max_w=6

logging.basicConfig(filename=f'log{max_w}', level=logging.INFO)

thread_local = threading.local()

q_in = queue.Queue()
q_out = queue.Queue()

model = ultralytics.YOLO("./weights/best.pt")

session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=30, pool_maxsize=30)
session.mount("https://", adapter)


def fun():
    
    
    #img_limit = settings.img_limit
    #res_limit = settings.res_limit
    
        
    ok = open("ok.csv", "w")
    failed = open("failed.csv", "w")
    
    h = "Исходное имя файла|Тип объекта|Размер (мм)|Число объектов|Путь к файлу\n"
    
    ok.write(h)
    failed.write(h)
    
    num = 0

    with ThreadPoolExecutor(max_workers=max_w) as executor:
        while(num < 200):
            response = sqs.receive_message(
                    QueueUrl=QUEUE_URL,
                    MaxNumberOfMessages=10,
                    VisibilityTimeout=60,
                    WaitTimeSeconds=0,
                    ReceiveRequestAttemptId='string'
                )
        
            
            messages = response.get("Messages", [])
            m1 = [ (json.loads(m["Body"]), m["ReceiptHandle"]) for m in messages]
            messages = [ (a["path"], a["filename"], b) for a,b in m1]
        
        
            num += len(messages)
            for message in messages:
                executor.submit(download_part, message, q_in, session)

            try:
                #here it will sleep and wait for task
                 res = model_part(q_in, q_out, model, ok, failed,
                                  session, wait=True)
                 executor.submit(upload_part, q_out, sqs, QUEUE_URL, session)

            except Exception as e:
                 print(f"An error occurred: {e}")
                 traceback.print_exc()
        
        
        while True:
                try:
                    res = model_part(q_in, q_out, model, ok, failed, session, wait=False)

                    if res == 1:
                        break

                    executor.submit(upload_part, q_out, sqs, QUEUE_URL, session)



                except Exception as e:
                    print(f"An error occurred: {e}")

        ok.close()
        failed.close()
        
    
    #if img_counter > img_limit:
    #    print("Should call next instance!", flush=True)
    #    return 0
    #else:
    #    print("Should finish!", flush=True)
    #    return 1


if __name__ == '__main__':
    fun()
    #if res == 0:
    #    sys.exit(10)
    #else:
    #    sys.exit(0)

