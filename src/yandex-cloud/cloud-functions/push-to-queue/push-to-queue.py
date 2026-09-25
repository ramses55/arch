#requirements.txt
#aioboto3
#aiolimiter

import os
import requests
import json
import aioboto3
import asyncio
    
from datetime import datetime
from itertools import islice
#from aiolimiter import AsyncLimiter

import time 

def handler(event, context):
    oauth_token = os.getenv("oauth_token")
    queue_url = os.getenv("queue_url")
    access_key_id = os.getenv('access_key_id')
    access_key = os.getenv('access_key')

    
    #limiter = AsyncLimiter(50, 1)
    
    
    sem = asyncio.Semaphore(16)

    
    
    def chunks(iterable, size):
        it = iter(iterable)
        while batch := list(islice(it, size)):
            yield batch
    
    async def send_batch(client, batch):
        #start = time.time()


        async with sem:
            #async with limiter:
            await client.send_message_batch(
                QueueUrl=queue_url,
                Entries=[
                    {"Id": str(i), "MessageBody":  json.dumps(msg)}
                    for i, msg in enumerate(batch)
                ]
            )
        #print("Batch took", time.time() - start)
    
    

    async def main():
        session = aioboto3.Session()
    
        async with session.client(
            "sqs",
            endpoint_url="https://message-queue.api.cloud.yandex.net",
            region_name="ru-central1",
            aws_access_key_id=access_key_id,
            aws_secret_access_key=access_key,
        ) as client:
    
            tasks = [
                send_batch(client, batch)
                for batch in chunks(files, 10)
            ]
    
            if (n>1000):
                await asyncio.gather(*tasks)
            else:
                for batch in chunks(files, 10):
                    await send_batch(client, batch)
    
    
    
    
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
            #print(f'latest--{ls.latest}')
            #print(len(items))
            #time.sleep(0.1) 
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
                    #print(epoch)
                    if epoch > ls.latest:
                        if epoch - ls.latest > 10:
                            ls.old = []
                        ls.latest = epoch - 1
                        if item['name'] not in ls.old:
                            files.append((item['path'], item['name']))
                            ls.old.append(item["name"])
                            #print(item['name'])
            offset += len(items)
    
    
        return files
    
    
    
    
    start = time.time()
    files = ls("")
    print("ls took", time.time() - start)
    n=len(files)
    

    start = time.time()
    asyncio.run(main())
    print("async took", time.time() - start)

    start = time.time()
    mkdir('ok')
    mkdir('failed')
    mkdir('ok/orig')
    mkdir('ok/marked')
    mkdir('failed/marked')
    mkdir('failed/orig')
    mkdir('completely-failed')
    print("num of files:", len(files))
    print("7 mkdirs took", time.time() - start)

    return {"body": {"num_mes": len(files)}}



if __name__ == "__main__":
    handler(1,2)
