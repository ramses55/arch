#also needs requirements.txt:
#boto3
#mysql-connector-python
#mysqlclient


import json
import boto3
import os
from mysql.connector import pooling

pool = pooling.MySQLConnectionPool(
    pool_name="pool1",
    pool_size=5,
    host="rc1b-ojtel5k6c967smm2.mdb.yandexcloud.net",
    port = 3306,
    user="user1",
    password= os.environ.get('DB_PASS'),
    database="db1"
)



QUEUE_URL = "https://message-queue.api.cloud.yandex.net/b1gn2cep80rfd8mos87v/dj60000000ifp1rm0516/queue"

sqs = boto3.client(
    "sqs",
    endpoint_url="https://message-queue.api.cloud.yandex.net",
    region_name="ru-central1"
)

def handler(event, context):

    conn = pool.get_connection()
    cursor = conn.cursor()

    insert_query = "REPLACE INTO data (file_name, status) VALUES (%s, %s)"


    for message in event["messages"]:
        bucket = message["details"]["bucket_id"]
        key = message["details"]["object_id"]

        task = {"bucket": bucket, "key": key}

        sqs.send_message(
            QueueUrl=QUEUE_URL,
            MessageBody=json.dumps(task)
        )
        values = (key, "IN QUEUE")
        cursor.execute(insert_query, values)
        conn.commit()

    cursor.close()
    conn.close()  
    return {"status": "ok"}
