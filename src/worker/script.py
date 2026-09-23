import boto3
import traceback
import time
import json

from utils import settings
from utils import work, download_b, mv
from utils import upload_part, model_part, download_part, make_csv

import sys
import os

import threading
from concurrent.futures import ThreadPoolExecutor
from itertools import count


import queue
import cv2
import numpy as np


#this prevents thread thrashing 
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import onnxruntime as ort


import requests
import logging


oauth_token = settings.oauth_token
access_key_id = settings.access_key_id
access_key = settings.access_key
QUEUE_URL = settings.queue_url
img_limit = settings.img_limit

sqs = boto3.client(
    "sqs",
    endpoint_url="https://message-queue.api.cloud.yandex.net",
    region_name="ru-central1",
    aws_access_key_id=access_key_id,
    aws_secret_access_key=access_key
)


s3 = boto3.client(service_name='s3',
                         endpoint_url='https://storage.yandexcloud.net',
                         aws_access_key_id=access_key_id,
                         aws_secret_access_key=access_key)



# max number of workers
max_w=5

logging.basicConfig(filename=f'log{max_w}', level=logging.INFO)

thread_local = threading.local()

q_in = queue.Queue()
q_out = queue.Queue()
current_item = None


opts = ort.SessionOptions()

# otherway onnx runs slower because of thread thrashing
opts.intra_op_num_threads = 1
opts.inter_op_num_threads = 1

# prevent global thread pool scaling issues
opts.use_per_session_threads = True



onnx = ort.InferenceSession(
    "./weights/new-weight.onnx",
    sess_options=opts,
    providers=["CPUExecutionProvider"]
)



session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=40, pool_maxsize=40)
session.mount("https://", adapter)


def fun():
    
    #res_limit = settings.res_limit
    
        
    
    # limits number of processed messages 
    num = 0
    # limits number of times while will iterate max
    i = 0 

    with ThreadPoolExecutor(max_workers=max_w) as executor:
        while(num < img_limit and i < 10):
            response = sqs.receive_message(
                    QueueUrl=QUEUE_URL,
                    MaxNumberOfMessages=10,
                    VisibilityTimeout=20,
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
                # here it will sleep and wait for task
                 res = model_part(q_in, q_out, onnx, ok, failed,
                                  session, wait=True)
                 if res == 0:
                    executor.submit(upload_part, q_out, sqs, QUEUE_URL, session)

            except Exception as e:
                 path, filename, receipt_handle, image = current_item
                 print(f"An error occurred in model_part {filename}: {e}")
                 print("=============================")
                 traceback.print_exc()
                 print("=============================")
                 mv(path, "completely-failed/" + filename)
                 q_in.task_done()
                 sqs.delete_message(QueueUrl=QUEUE_URL, ReceiptHandle=receipt_handle)

        
        
        while q_in.qsize() != 0:
                try:
                    res = model_part(q_in, q_out, onnx, ok, failed, session, wait=False)
                    print("res", res)

                    print("q_in size:", q_in.qsize())

                    if res == 0:
                        executor.submit(upload_part, q_out, sqs, QUEUE_URL, session)
                        print("q_in size:", q_in.qsize(), "pushed to executor")


                    if res == 1 or q_in.qsize() == 0:
                        break




                except Exception as e:
                    path, filename, receipt_handle, image = current_item
                    print(f"An error occurred in model_part {filename}: {e}")
                    print("=============================")
                    traceback.print_exc()
                    print("=============================")
                    mv(path, "completely-failed/" + filename)
                    q_in.task_done()
                    sqs.delete_message(QueueUrl=QUEUE_URL, ReceiptHandle=receipt_handle)
                    #print(f"An error occurred in upload_part: {e}")

        executor.shutdown(wait=True)




ok = open("ok.csv", "w")
failed = open("failed.csv", "w")

h = "Исходное имя файла|Тип объекта|Размер (мм)|Число объектов|Путь к файлу\n"

ok.write(h)
failed.write(h)


for i in range(3):
    print(f"Started fun(): {i}")
    fun()

ok.close()
failed.close()


make_csv("ok", s3, session)
make_csv("failed", s3, session)
print("Made CSVs");
