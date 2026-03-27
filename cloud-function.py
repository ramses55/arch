#also needs requirements.txt
import json
import boto3

QUEUE_URL = "https://message-queue.api.cloud.yandex.net/b1gn2cep80rfd8mos87v/dj60000000ifp1rm0516/queue"

sqs = boto3.client(
    "sqs",
    endpoint_url="https://message-queue.api.cloud.yandex.net",
    region_name="ru-central1"
)

def handler(event, context):

    for message in event["messages"]:
        bucket = message["details"]["bucket_id"]
        key = message["details"]["object_id"]

        task = {"bucket": bucket, "key": key}

        sqs.send_message(
            QueueUrl=QUEUE_URL,
            MessageBody=json.dumps(task)
        )

    return {"status": "ok"}
