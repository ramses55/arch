#also needs requirements.txt
#boto3


import boto3
import os


def handler(event, context):
    access_key_id = os.getenv('access_key_id')
    access_key = os.getenv('access_key')
    queue_url = os.getenv('queue_url')
    
    sqs = boto3.client(
        "sqs",
        endpoint_url="https://message-queue.api.cloud.yandex.net",
        region_name="ru-central1",
        aws_access_key_id=access_key_id,
        aws_secret_access_key=access_key
    )
    
    #print(sqs.list_queues())
    response = sqs.get_queue_attributes(
            QueueUrl = queue_url,
            AttributeNames = ['ApproximateNumberOfMessages',
                              'ApproximateNumberOfMessagesDelayed',
                              'ApproximateNumberOfMessagesNotVisible'
                              ]
            )
    
    s = 0
    attr = response.get('Attributes', [])
    print(attr)
    for k,v in attr.items():
        s += int(v)
    
    return {"sum": s}

