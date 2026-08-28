import cv2
import numpy as np
import os
import time

from .onnx_result import onnx_result, preprosses


import json
import io
from utils import ls, mv, download, upload, make_csv, download_b
import queue

import time
import logging



logging.basicConfig(level=logging.INFO)



def download_part(message, q, session):
    t0 = time.perf_counter()
    cpu0 = time.thread_time()
    path, filename, receipt_handle = message

    res = download_b(path, session)

    if type(res) is int:
        print(f"Error download: {res}. {path}")
        #mv(path, "failed/orig/")
        

    else:
        image = cv2.imdecode(np.frombuffer(res, np.uint8), cv2.IMREAD_COLOR)

        queue_el = (path, filename, receipt_handle, image)
        q.put(queue_el)


    dt = time.perf_counter() - t0;
    dcpu = time.thread_time() - cpu0;
    #logging.info("Download %s took wall=%.3f s cpu=%.3f s", filename, dt, dcpu)



def model_part(queue_in, queue_out, onnx, file_ok, file_failed, session, wait):
    t0 = time.perf_counter()
    cpu0 = time.thread_time()

    try:
        if wait:
            path, filename, receipt_handle, image = queue_in.get(timeout=5)
        else:
            path, filename, receipt_handle, image = queue_in.get_nowait()

    except queue.Empty:
        return 1

    prep_res = preprosses(image)
    onnx_res0 = onnx.run(None, {'images': prep_res})[0][0]
    o = onnx_result(onnx_res = onnx_res0, filename = filename,
                orig_image = image, new_shape = onnx.get_inputs()[0].shape,
                use_nms = True, conf = 0.05, thres = 0.4)       

    #print(filename)
    #output = model(image,
    #            conf=0.25,
    #            save=False,
    #            show=False,
    #            verbose=False)
    
    #r = result(output[0], filename)
    t01 = time.perf_counter()
    cpu01 = time.thread_time()
    o.all(session)
    dt = time.perf_counter() - t01;
    dcpu = time.thread_time() - cpu01;
    #logging.info("OCR %s took wall=%.3f s cpu=%.3f s", filename, dt, dcpu)
    m = o.csv_res()
    buffer = o.draw()
    ext = filename.split('.')[-1]
        
        
        
    #checks if OCR worked correctly
    if not o.file_name or 'd!!!:' in o.file_name[0]  or 'ind' in o.file_name[0] or len(o.names) == 0:
        new_path = "disk:/Приложения/arch_fragments/failed/marked/"+filename
        r = mv(path, "failed/orig/" + filename)
        if (r // 100 != 2):
            print("Move status:", r, filename)

        file_failed.write(m + '\n')
    else:
        new_file_name =  o.file_name[0] + '.' + ext
        new_path ="disk:/Приложения/arch_fragments/ok/marked/" + new_file_name
        #print("mv code:", mv(path, "ok/orig/" + new_file_name))
        r = mv(path, "ok/orig/" + new_file_name)
        if (r // 100 != 2):
            print("Move status:", r, new_file_name)

        file_ok.write(m + '\n')

    queue_el = (new_path, receipt_handle, buffer)
    queue_out.put(queue_el)
    queue_in.task_done()
    #executor.submit(upload_part, q_out, sqs, QUEUE_URL, session)

    dt = time.perf_counter() - t0;
    dcpu = time.thread_time() - cpu0;
    #logging.info("Image process %s took wall=%.3f s cpu=%.3f s", filename, dt, dcpu)
    return 0


def upload_part(q,sqs, queue_url, session):
    t0 = time.perf_counter()
    cpu0 = time.thread_time()

    new_path, receipt_handle, buffer = q.get()
    r=upload(buffer, new_path, session)
    if (r // 100 != 2):
        print("Upload status:", r, new_path)

    sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)

    q.task_done()

    dt = time.perf_counter() - t0;
    dcpu = time.thread_time() - cpu0;
    #logging.info("Image upload %s took wall=%.3f s cpu=%.3f s", new_path, dt, dcpu)
