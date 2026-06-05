import ultralytics
import cv2
import os
import time
from utils import result
import json
from mysql.connector import pooling
import io
from utils import ls, mv, download, upload, make_csv





def work(path, filename, pool):
    '''
    This function returns 0 if work is done succesfully and 1 overwise
    '''



    if filename == '-' and path == '-':
        conn = pool.get_connection()
        make_csv(conn, True)
        make_csv(conn, False)
        conn.close()
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
        
        #time.sleep(0.5)
        print(local_name)
        output = model(local_name,
                    conf=0.1,
                    save=False,
                    show=False,
                    verbose=False)
        
        r = result(output[0], filename)
        r.all()
        m = r.csv_res()
        buffer = r.draw()
        
        
        
        conn = pool.get_connection()
        cursor = conn.cursor()
        
        insert_query = """
                            INSERT INTO data (file_name, status, result, new_file_name)
                            VALUES (%s, %s, %s, %s)
                            ON DUPLICATE KEY UPDATE result = VALUES(result)
                        """
        
        #checks if OCR worked correctly
        if not r.file_name or r.file_name[0] == 'OCR failed!' or 'index' in r.file_name[0] or len(r.names) == 0 :
            new_path = "disk:/Приложения/arch_fragments/failed/marked/"+filename
            upload(buffer, new_path)
            mv(path, "failed/orig/" + filename)
            values = (filename, "FAILED", m, filename)
        else:
            new_file_name =  r.file_name[0] + '.' + ext
            values = (filename, "DONE", m, new_file_name)
            new_path ="disk:/Приложения/arch_fragments/ok/marked/" + new_file_name
            print("upload code:", upload(buffer, new_path))
            print("mv code:", mv(path, "ok/orig/" + new_file_name))
        
        cursor.execute(insert_query, values)
        conn.commit()
        
        cursor.close()
        conn.close()  

        return 0
