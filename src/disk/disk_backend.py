from utils_disk import push_to_queue, ls, mkdir, settings
import requests

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
print(files)
#if files:
#    push_to_queue([['-','-']])



try:
    requests.post(worker_url, timeout=0.1)
except requests.exceptions.Timeout:
    print("Triggered the worker instance")

