import requests
import time
import boto3
import json
from datetime import datetime



with open("../keys/oauth_token") as f:
    oauth_token = f.readline().rstrip()



with open("../keys/s3-key-id", "r") as f:
    access_key = f.readline().rstrip()

with open("../keys/s3-key", "r") as f:
    secret_key = f.readline().rstrip()


QUEUE_URL = "https://message-queue.api.cloud.yandex.net/b1gn2cep80rfd8mos87v/dj60000000ifp1rm0516/queue"

sqs = boto3.client(
    "sqs",
    endpoint_url="https://message-queue.api.cloud.yandex.net",
    region_name="ru-central1",
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key
)


def folder_path():
    url = "https://cloud-api.yandex.net/v1/disk/resources?path=app:/"
    headers = {"Authorization": f'OAuth {oauth_token}'}
    r = requests.get(url=url, headers = headers)

    print(r.json())




def mkdir(dirname: str):
    '''
        This function creates folder if one with same name does not already
        exist. It creates folder "disk:/Приложения/arch_fragments/{dirname}". If
        returned status code is 409 directory already exists.

        Args:
            dirname (str): name of directory to create

        Returns:
            status_code (int): status_code
            

    '''
    url = "https://cloud-api.yandex.net/v1/disk/resources"
    headers = {"Authorization": f'OAuth {oauth_token}'}
    #params = { 'path': , 'fields': }
    params = { 'path': f'disk:/Приложения/arch_fragments/{dirname}'}

    r = requests.put(url = url,
                     params = params,
                     headers = headers
                     )
    print(r.json())
    return r.status_code


def ls(dirname: str,
       limit: int = 999
       ):
    '''
        This function list files in the directory
        "disk:/Приложения/arch_fragments/{dirname}" which it has not seen
        before.

        Args:
            dirname (str): name of directory 

        Returns:
            files (list): list of files
            

    '''
    url = "https://cloud-api.yandex.net/v1/disk/resources"
    headers = {"Authorization": f'OAuth {oauth_token}'}
    offset = 0
    files = []
    if not hasattr(ls, "latest"):
            ls.latest = 0

    items = ['l']

    while (len(items) != 0):
        #print(f'latest--{ls.latest}')
        time.sleep(0.1) 
        params = {
                'path':  f'disk:/Приложения/arch_fragments/{dirname}',
                'fields':
                '_embedded.items.path,_embedded.items.type,_embedded.items.name,_embedded.items.modified',
                'limit': f'{limit}',
                'offset': f'{offset}',
                'sort': 'modified'
                }

        r = requests.get(url=url,
                        headers=headers,
                        params=params
                        )

        items = r.json()['_embedded']['items']
        for item in items:
            if item['type'] == 'file':
                epoch = int(datetime.fromisoformat(item['modified']).timestamp())
                #print(epoch)
                if epoch > ls.latest:
                    ls.latest = epoch
                    files.append((item['path'], item['name']))
        offset += len(items)


    return files



def push_to_queue(files: list):
    '''
        This function pushes files to sqs-like Message Queue

        Args:
            files (list): list of tuples (filepath, filename)
    '''
    for file in files:
        task = {"path": file[0], "filename": file[1]}
        
        sqs.send_message(
            QueueUrl=QUEUE_URL,
            MessageBody=json.dumps(task)
        )


def mv(file,
       dirname: str
       ):
    '''
        This function moves file from its old path to
        'disk:/Приложения/{dirname}/{filename}'
    '''
    pass
