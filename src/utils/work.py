import ultralytics
import cv2
import numpy as np
import os
import time
from utils import result
import json
import io
from utils import ls, mv, download, upload, make_csv, download_b
import queue





def work(path, filename, file_ok, file_failed, ln):
    '''
    This function returns 0 if work is done succesfully and 1 otherwise
    '''



    if filename == '-' and path == '-':
        make_csv("ok")
        make_csv("failed")
        return 0

    else:

        local_name = f"image-{ln}"    
        code=download(path, local_name)
        if code != 200:
            print(f"Error download: {code}")
            mv(path, "failed/orig/")

            return 1
    
        ext = filename.split('.')[-1]
        local_name = f"image-{ln}.{ext}"    
    
    
        model = ultralytics.YOLO("./weights/best.pt")
        
        #print(local_name)
        output = model(local_name,
                    conf=0.25,
                    save=False,
                    show=False,
                    verbose=False)
        
        r = result(output[0], filename)
        r.all()
        m = r.csv_res()
        buffer = r.draw()
        
        
        
        
        #checks if OCR worked correctly
        if not r.file_name or 'OCR failed!' in r.file_name[0]  or 'ind' in r.file_name[0] or len(r.names) == 0:
            new_path = "disk:/Приложения/arch_fragments/failed/marked/"+filename
            upload(buffer, new_path)
            mv(path, "failed/orig/" + filename)
            #values = (filename, "FAILED", m, filename)
            #file_failed.write(m + '\n')
        else:
            new_file_name =  r.file_name[0] + '.' + ext
            #values = (filename, "DONE", m, new_file_name)
            new_path ="disk:/Приложения/arch_fragments/ok/marked/" + new_file_name
            print("upload code:", upload(buffer, new_path))
            print("mv code:", mv(path, "ok/orig/" + new_file_name))
            #file_ok.write(m + '\n')
        

        return 0





def download_part(message, q, session):
    path, filename, receipt_handle = message

    res = download_b(path, session)

    if type(res) is int:
        print(f"Error download: {res}")
        mv(path, "failed/orig/")
        

    else:
        image = cv2.imdecode(np.frombuffer(res, np.uint8), cv2.IMREAD_COLOR)

        queue_el = (path, filename, receipt_handle, image)
        q.put(queue_el)



def model_part(queue_in, queue_out, model, file_ok, file_failed):

    try:
        path, filename, receipt_handle, image = queue_in.get(timeout=0.5)

    except queue.Empty:
        return 1

            

    print(filename)
    output = model(image,
                conf=0.25,
                save=False,
                show=False,
                verbose=False)
    
    r = result(output[0], filename)
    r.all()
    m = r.csv_res()
    buffer = r.draw()
    ext = filename.split('.')[-1]
        
        
        
    #checks if OCR worked correctly
    if not r.file_name or 'OCR failed!' in r.file_name[0]  or 'ind' in r.file_name[0] or len(r.names) == 0:
        new_path = "disk:/Приложения/arch_fragments/failed/marked/"+filename
        mv(path, "failed/orig/" + filename)
        file_failed.write(m + '\n')
    else:
        new_file_name =  r.file_name[0] + '.' + ext
        new_path ="disk:/Приложения/arch_fragments/ok/marked/" + new_file_name
        print("mv code:", mv(path, "ok/orig/" + new_file_name))
        file_ok.write(m + '\n')

    queue_el = (new_path, receipt_handle, buffer)
    queue_out.put(queue_el)
    queue_in.task_done()
    return 0


def upload_part(q,sqs, queue_url, session):
    new_path, receipt_handle, buffer = q.get()
    print("upload code:", upload(buffer, new_path, session))
    sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)

    q.task_done()
        
