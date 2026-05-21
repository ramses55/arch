import boto3
import ultralytics
import cv2
import os
import time
from utils import result
import json
from mysql.connector import pooling
import io
from utils import ls, mv, download, upload, make_csv


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
        print("Got a message!")
        #print(response)
    else:
        print("No messages yet!")
        continue



    for message in response["Messages"]:
        body_info = json.loads(message["Body"])
        path = body_info["path"]
        filename = body_info["filename"]
        receipt_handle = message['ReceiptHandle']

        if filename == '-' and path == '-':
            conn = pool.get_connection()
            make_csv(conn, True)
            make_csv(conn, False)
            response_del = sqs.delete_message(QueueUrl=QUEUE_URL,
                                             ReceiptHandle=receipt_handle)
            conn.close()
        else:

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
            
            r = result(output[0], filename)
            r.all()
            m = r.csv_res()
            buffer = r.draw()
            
            
            
            conn = pool.get_connection()
            cursor = conn.cursor()
            
            insert_query = """
                                INSERT INTO data (file_name, status, result)
                                VALUES (%s, %s, %s)
                                ON DUPLICATE KEY UPDATE result = VALUES(result)
                            """
            
            #checks if OCR worked correctly
            if r.file_name[0] == 'OCR failed!' or 'index' in r.file_name[0] or len(r.names) == 0 :
                new_path = "disk:/Приложения/arch_fragments/failed/marked/"+filename
                upload(buffer, new_path)
                mv(path, "failed/orig")
                values = (filename, "FAILED", m)
            else:
                values = (filename, "DONE", m)
                new_path = "disk:/Приложения/arch_fragments/ok/marked/"+filename
                upload(buffer, new_path)
                mv(path, "ok/orig")
            
            cursor.execute(insert_query, values)
            conn.commit()
            
            cursor.close()
            conn.close()  


            #deletes message from queue
            response_del = sqs.delete_message(QueueUrl=QUEUE_URL,
                                             ReceiptHandle=receipt_handle)
