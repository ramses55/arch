from utils import push_to_queue, ls
import time




while True:
    time.sleep(5)
    files = ls("")
    push_to_queue(files)
