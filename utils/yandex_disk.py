import requests
import time
import boto3
import json
import os
from datetime import datetime



with open("./keys/oauth_token") as f:
    oauth_token = f.readline().rstrip()



with open("./keys/s3-key-id", "r") as f:
    access_key = f.readline().rstrip()

with open("./keys/s3-key", "r") as f:
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
    if not hasattr(ls, "old"):
        ls.old = []

    items = ['l']

    while (len(items) != 0):
        print(f'latest--{ls.latest}')
        print(len(items))
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

        if r.status_code != 200:
            break
        items = r.json()['_embedded']['items']
        for item in items:
            if item['type'] == 'file':
                epoch = int(datetime.fromisoformat(item['modified']).timestamp())
                print(epoch)
                if epoch > ls.latest:
                    if epoch - ls.latest > 10:
                        ls.old = []
                    ls.latest = epoch - 1
                    if item['name'] not in ls.old:
                        files.append((item['path'], item['name']))
                        ls.old.append(item["name"])
                        print(item['name'])
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



def download(path: str):
    '''
        This function downloads the file

        Args:
            path (str): path of a file to be downloaded
        
        Returns:
            exit_codes (tuple): (exit_code1, exit_code2)
    '''

    url = "https://cloud-api.yandex.net/v1/disk/resources/download"
    headers = {"Authorization": f'OAuth {oauth_token}'}
    params = {'path': f'{path}'}

    r = requests.get(url = url,
                     headers = headers,
                     params = params
                     )


    ext = path.split(".")[-1]
    if r.status_code == 200:
        download_url = r.json()['href']
        r1 = requests.get(url = download_url)
        if r1.status_code == 200:
            with open(f"image.{ext}", "wb") as f:
                f.write(r1.content)
        else:
            return r1.status_code
    else:
        return r.status_code

    return 200




def upload(buffer,
           path: str):
    '''
        This function uploads the file

        Args:
            filename (str): local file name
            path (str): path of a file to be uploaded


        Returns:
            exit_codes (tuple): (exit_code1, exit_code2)
    '''


    url = "https://cloud-api.yandex.net/v1/disk/resources/upload"
    headers = {"Authorization": f'OAuth {oauth_token}'}
    params = {'path': f'{path}',
              'overwrite': 'true'
              }

    r = requests.get(url = url,
                     headers = headers,
                     params = params
                     )


    if r.status_code == 200:
        upload_url = r.json()['href']
        #with open(filename, "rb") as f:
        r1 = requests.put(url = upload_url, data=buffer)
        return r1.status_code
    else:
        return r.status_code








def mv(filepath: str,
       dirname: str
       ):
    '''
        This function moves file from its old path to
        'disk:/Приложения/arch_fragments/{dirname}/{filename}'

        Args:
            filepath (str): path of the file in yandex disk
            dirname (str): directory name to move file into

        Returns:
            status_code (int): status code
    '''
    filename = os.path.basename(filepath)
    path = f"disk:/Приложения/arch_fragments/{dirname}/{filename}"


    url = "https://cloud-api.yandex.net/v1/disk/resources/move"
    headers = {"Authorization": f'OAuth {oauth_token}'}
    params = {
                'from': f"{filepath}",
                'path': f'{path}',
                'overwrite': 'true',
                'force_async': 'true'
              }

    r = requests.post(url=url,
                     headers=headers,
                     params=params,
                     )

    #print(r.json())
    return r.status_code




