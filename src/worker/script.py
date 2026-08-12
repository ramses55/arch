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
access_key_id = settings.access_key_id
access_key = settings.access_key
QUEUE_URL = settings.queue_url
worker_url = settings.worker_url
img_limit = settings.img_limit

sqs = boto3.client(
    "sqs",
    endpoint_url="https://message-queue.api.cloud.yandex.net",
    region_name="ru-central1",
    aws_access_key_id=access_key_id,
    aws_secret_access_key=access_key
)

max_w=6

logging.basicConfig(filename=f'log{max_w}', level=logging.INFO)

thread_local = threading.local()

q_in = queue.Queue()
q_out = queue.Queue()

model = ultralytics.YOLO("./weights/best.onnx", task='obb')

session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=40, pool_maxsize=40)
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
    i = 0 #number of times while will iterate max

    with ThreadPoolExecutor(max_workers=max_w) as executor:
        while(num < img_limit and i < 10):
            response = sqs.receive_message(
                    QueueUrl=QUEUE_URL,
                    MaxNumberOfMessages=10,
                    VisibilityTimeout=60,
                    WaitTimeSeconds=0,
                    ReceiveRequestAttemptId='string'
                )
        
            
            messages = response.get("Messages", [])
            m1 = [ (json.loads(m["Body"]), m["ReceiptHandle"]) for m in messages]
            messages = [ (a[0], a[1], b) for a,b in m1]
        
        
            num += len(messages)
            if (len(messages) == 0):
                i+=1
                continue

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
        
        
        while q_in.qsize() != 0:
                try:
                    res = model_part(q_in, q_out, model, ok, failed, session, wait=False)

                    print("q_in size:", q_in.qsize())

                    if res == 1 or q_in.qsize() == 0:
                        break

                    executor.submit(upload_part, q_out, sqs, QUEUE_URL, session)



                except Exception as e:
                    print(f"An error occurred: {e}")

        ok.close()
        failed.close()
        
    


for i in range(3):
    print(f"Started fun(): {i}")
    fun()

