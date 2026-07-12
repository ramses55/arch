import sys
import json

from utils_disk import push_to_queue, ls, mkdir, settings

worker_url = settings.worker_url

#makes sure certain dirs do exist
mkdir('ok')
mkdir('failed')
mkdir('ok/orig')
mkdir('ok/marked')
mkdir('failed/marked')
mkdir('failed/orig')

files = ls("")
push_to_queue(files)
print("num of files:", len(files))

with open('/tmp/output.json', "w") as f:
    json.dump({'result': 'ok'}, f)
