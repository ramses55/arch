import requests
import time
import boto3
import json
import csv
from datetime import datetime
from utils.config import settings
import openpyxl

from botocore.exceptions import ClientError



oauth_token = settings.oauth_token
access_key = settings.access_key_id
secret_key = settings.access_key
QUEUE_URL = settings.queue_url

#sqs = boto3.client(
#    "sqs",
#    endpoint_url="https://message-queue.api.cloud.yandex.net",
#    region_name="ru-central1",
#    aws_access_key_id=access_key,
#    aws_secret_access_key=secret_key
#)
#
#
#
#
#s3 = boto3.client(service_name='s3',
#                         endpoint_url='https://storage.yandexcloud.net',
#                         aws_access_key_id=access_key,
#                         aws_secret_access_key=secret_key)
#

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


def ls_db(dirname: str,
       limit: int = 999
       ):
    '''
        This function fetches data for db and list files it did not see before.
        It uses ls but gives it state.


        Args:
            dirname (str): name of directory 

        Returns:
            files (list): list of files

    '''


    try:
        print("Trying to connect!")
        conn = mysql.connector.connect(
            port=3306,
            host=settings.mysql_host,
            user=settings.mysql_user,
            password=settings.mysql_pass,
            database=settings.mysql_db,
            connect_timeout=5
        )
    
        if conn.is_connected():
            print("Successfully connected to the database")
            
            cur = conn.cursor()
            
            
            get_ls_state = '''
                            SELECT new_file_name
                            FROM data
                            WHERE file_name = "state_for_ls"
                           '''
    
            put_ls_state = '''
                                INSERT INTO data (file_name, new_file_name)
                                VALUES (%s, %s)
                                ON DUPLICATE KEY update new_file_name = VALUES(new_file_name)
                           '''

            
            print("Get state")
            cur.execute(get_ls_state)

            res = cur.fetchall()
            state=int(res[0][0])
            print(state)

            fun = ls
            fun.latest = state
            files = fun(dirname, limit)

            state = fun.latest


            print("Push state")
            cur.execute(put_ls_state, ('state_for_ls',str(state)))
            conn.commit()
            res = cur.fetchall()

            return files




    except Error as e:
        print(f"Error while connecting to MySQL server: {e}")
    
    finally:
        if 'connection' in locals() and conn.is_connected():
            cur.close()
            conn.close()
            print("MySQL connection is closed")
    





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



def push_to_queue(files: list):
    '''
        This function pushes files to sqs-like Message Queue

        Args:
            files (list): list of tuples (filepath, filename)
    '''
    for file in files:
        task = {'path': file[0], 'filename': file[1]}
        
        sqs.send_message(
            QueueUrl=QUEUE_URL,
            MessageBody=json.dumps(task)
        )


def download_b(path: str, session):
    '''
        This function downloads the file and returns its byte representation

        Args:
            path (str): path of a file to be downloaded
        
        Returns:
            exit_codes (tuple): (exit_code1, exit_code2)
    '''

    url = "https://cloud-api.yandex.net/v1/disk/resources/download"
    headers = {"Authorization": f'OAuth {oauth_token}'}
    params = {'path': f'{path}'}

    r = session.get(url = url,
                     headers = headers,
                     params = params
                     )


    if r.status_code == 200:
        download_url = r.json()['href']
        r1 = session.get(url = download_url)
        if r1.status_code == 200:
            return  r1.content
        else:
            return r1.status_code
    else:
        return r.status_code




def download(path: str, local_name: str):
    '''
        This function downloads the file

        Args:
            path (str): path of a file to be downloaded
            local_name (str): local_name of downloaded file
        
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
            with open(f"{local_name}.{ext}", "wb") as f:
                f.write(r1.content)
        else:
            return r1.status_code
    else:
        return r.status_code

    return 200




def upload(buffer,
           path: str,
           session):
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

    r = session.get(url = url,
                     headers = headers,
                     params = params
                     )


    if r.status_code == 200:
        upload_url = r.json()['href']
        r1 = session.put(url = upload_url, data=buffer)
        return r1.status_code
    else:
        return r.status_code








def mv(filepath: str,
       newpath: str
       ):
    '''
        This function moves file from its old path to
        'disk:/Приложения/arch_fragments/{newpath}'

        Args:
            filepath (str): path of the file in yandex disk
            dirname (str): directory name to move file into

        Returns:
            status_code (int): status code
    '''
    #filename = os.path.basename(filepath)
    path = f"disk:/Приложения/arch_fragments/{newpath}"


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

    return r.status_code



def ls_s(dirname: str,
       limit: int = 999
       ):
    '''
        This function list files in the directory
        "disk:/Приложения/arch_fragments/{dirname}", simple version of ls.
        Different function is used to eliminate work with ls attributes like 
        .old and .latest


        Args:
            dirname (str): name of directory 

        Returns:
            files (list): list of files
            

    '''
    url = "https://cloud-api.yandex.net/v1/disk/resources"
    headers = {"Authorization": f'OAuth {oauth_token}'}
    offset = 0
    files = []

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
                files.append((item['path'], item['name']))

        offset += len(items)


    return files

def make_csv(st, s3, session):

    merged = dict()
    header = ["Исходное имя файла", "Тип объекта", "Размер (мм)", "Число объектов", "Путь к файлу"]

    try:
        s3.download_file("for-csv", f"{st}-r.csv", f"{st}-r.csv")

        #it is important that older file goes first
        with open(f"{st}-r.csv", "r") as f:
            old = csv.reader(f, delimiter = ',')

            try:

                next(old)

                for row in old:
                    merged[row[0]] = row
            except StopIteration:
                #print("Got stop iter old")
                pass

    except ClientError:
        pass


    with open(f"{st}.csv", "r") as f:
        new = csv.reader(f, delimiter = '|')

        try:

            next(new)

            for row in new:
                #will update already present rows with new values
                merged[row[0]] = row
        except StopIteration:
            #print("Got stop iter new")
            pass

    data = [merged[k] for k in merged]
    data = [header] + data
    with open(f"{st}-res.csv", "w") as f:
        writer = csv.writer(f, delimiter = ",")
        writer.writerows(data)

    s3.upload_file(f"{st}-res.csv", "for-csv", f"{st}-r.csv")


    with open(f"{st}-res.csv", "rb") as f:
           buffer = f.read()

    path = f"disk:/Приложения/arch_fragments/{st}/{st}.csv"
    code = upload(buffer, path, session)

    wb = openpyxl.Workbook()
    ws = wb.active

    for row in data:
        ws.append(row)

    wb.save(f"{st}.xlsx")


    with open(f"{st}.xlsx", "rb") as f:
           buffer = f.read()

    path = f"disk:/Приложения/arch_fragments/{st}/{st}.xlsx"
    code = upload(buffer, path, session)

    return code

def make_csv_db(conn, suc=True):

    if suc:
        files = ls_s('ok/orig')
        files = [f[1] for f in files]

    if not suc:
        files = ls_s('failed/orig')
        files = [f[1] for f in files]

    if not files:
        return 0

    placeholder = ", ".join(["%s"] * len(files))

    get_table_ok = f'''
                    SELECT result
                    FROM data
                    WHERE status = %s AND new_file_name IN ({placeholder})
                '''



    get_table_failed = f'''
                    SELECT result
                    FROM data
                    WHERE status = %s AND file_name IN ({placeholder})
                '''
    

    cur = conn.cursor()
    if suc:
        filename = "ok.csv"
        status = "DONE"
        path = 'ok/ok.csv'
        cur.execute(get_table_ok, [status] + files)
    else:
        filename = "failed.csv"
        status = "FAILED"
        path = 'failed/failed.csv'
        cur.execute(get_table_failed, [status] + files)


    
    with open(filename, 'w', newline="") as f:
        writer = csv.writer(f, delimiter = ",")
    
        writer.writerow(["Исходное имя файла", "Тип объекта", "Размер (мм)",
                         "Загрязнение", "Обугливание", "Число объектов",
                         "Путь к файлу"])
    
    
        while True:
            rows = cur.fetchmany(1000)
    
            if not rows:
                break
    
            #flattens rows
            new_rows = [c for b in rows for c in b]
            for row in new_rows:
                values = row.replace(",", ";").split("|")
                writer.writerow(values)


    with open(filename, "rb") as f:
        buffer = f.read()

    path = "disk:/Приложения/arch_fragments/" + path
    code = upload(buffer, path)
    print(code)

    cur.close()
    
    return 1
