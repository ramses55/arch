from utils import push_to_queue, ls_db, mkdir
import time


#makes sure certain dirs do exist
mkdir('ok')
mkdir('failed')
mkdir('ok/orig')
mkdir('ok/marked')
mkdir('failed/marked')
mkdir('failed/orig')


#push_to_queue([['-','-']])
while True:
    time.sleep(5)
    files = ls_db("")
    push_to_queue(files)
    print(files)
    if files:
        push_to_queue([['-','-']])
