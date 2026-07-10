from utils import settings
from utils import work
import boto3
import requests
import traceback
import time
import json




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
img_limit = 0

    
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
            WaitTimeSeconds=20,
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
            #time.sleep(0)
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
    if (res_counter > 4 or img_counter > img_limit):
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


#calls next instance
if img_counter > img_limit:
    try:
       requests.post(worker_url, timeout=0.1)
    except requests.exceptions.Timeout:
        print("Triggered the next instance")

