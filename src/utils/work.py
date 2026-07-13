import ultralytics
import cv2
import os
import time
from utils import result
import json
import io
from utils import ls, mv, download, upload, make_csv





def work(path, filename, file_ok, file_failed):
    '''
    This function returns 0 if work is done succesfully and 1 otherwise
    '''



    if filename == '-' and path == '-':
        make_csv("ok")
        make_csv("failed")
        return 0

    else:
        code=download(path)
        if code != 200:
            print(f"Error download: {code}")
            mv(path, "failed/orig/")

            return 1
    
        ext = filename.split('.')[-1]
        local_name = f"image.{ext}"    
    
    
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
            file_failed.write(m + '\n')
        else:
            new_file_name =  r.file_name[0] + '.' + ext
            #values = (filename, "DONE", m, new_file_name)
            new_path ="disk:/Приложения/arch_fragments/ok/marked/" + new_file_name
            print("upload code:", upload(buffer, new_path))
            print("mv code:", mv(path, "ok/orig/" + new_file_name))
            file_ok.write(m + '\n')
        

        return 0



