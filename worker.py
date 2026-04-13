import boto3
import ultralytics
import cv2
import os
import time
from utils import result
import json
from mysql.connector import pooling
import io
from utils import ls, mv, download, upload


with open("./keys/s3-key-id", "r") as f:
    access_key = f.readline().rstrip()

with open("./keys/s3-key", "r") as f:
    secret_key = f.readline().rstrip()


with open("./keys/mysql_pass", "r") as f:
    mysql_pass = f.readline().rstrip()

s3 = boto3.client(service_name='s3',
                         endpoint_url='https://storage.yandexcloud.net',
                         aws_access_key_id=access_key,
                         aws_secret_access_key=secret_key)



QUEUE_URL = "https://message-queue.api.cloud.yandex.net/b1gn2cep80rfd8mos87v/dj60000000ifp1rm0516/queue"

sqs = boto3.client(
    "sqs",
    endpoint_url="https://message-queue.api.cloud.yandex.net",
    region_name="ru-central1",
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key
)





pool = pooling.MySQLConnectionPool(
    pool_name="pool1",
    pool_size=5,
    host="rc1b-ojtel5k6c967smm2.mdb.yandexcloud.net",
    port = 3306,
    user="user1",
    password= mysql_pass,
    database="db1"
)
upload_bucket_name = "marked"

while (True):
    
    
    
    
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
        print(response)
    else:
        print("No messages yet!")
        continue



    for message in response["Messages"]:
        body_info = json.loads(message["Body"])
        path = body_info["path"]
        filename = body_info["filename"]
        receipt_handle = message['ReceiptHandle']
        code=download(path)
        if code != 200:
            print(f"Error: {code}")
            mv(path, "failed/orig")
            response_del = sqs.delete_message(QueueUrl=QUEUE_URL,
                                             ReceiptHandle=receipt_handle)
            break

        ext = filename.split('.')[-1]
        local_name = f"image.{ext}"    
    
    
        model = ultralytics.YOLO("./best.pt")
        
        time.sleep(0.5)
        print(local_name)
        output = model(local_name,
                    conf=0.1,
                    save=False,
                    show=False,
                    verbose=False)
        
        try:
            # Code that might fail
            r = result(output[0])
            r.all()
            m = r.csv_res()
            buffer = r.draw()
            new_path = "disk:/Приложения/arch_fragments/ok/marked/"+filename
            upload(buffer, new_path)
            mv(path, "ok/orig")

            #deletes message from queue
            response_del = sqs.delete_message(QueueUrl=QUEUE_URL,
                                             ReceiptHandle=receipt_handle)


            #conn = pool.get_connection()
            #cursor = conn.cursor()

            #update_query = """
            #                    UPDATE data
            #                    SET status = "FINISHED", result = %s
            #                    WHERE file_name = %s
            #                """

            #values = (m, filename)
            #cursor.execute(update_query, values)
            #conn.commit()

            #cursor.close()
            #conn.close()  
        except Exception as e:
            print(f"An error occurred: {e}")
            print(f"Error type: {type(e).__name__}")
        

            #conn = pool.get_connection()
            #cursor = conn.cursor()

            #update_query = """
            #                    UPDATE data
            #                    SET status = "FAILED", result = %s
            #                    WHERE file_name = %s
            #                """

            #values = (str(e), filename)
            #cursor.execute(update_query, values)
            #conn.commit()

            #cursor.close()
            #conn.close()  
            response_del = sqs.delete_message(QueueUrl=QUEUE_URL,
                                             ReceiptHandle=receipt_handle)
        
