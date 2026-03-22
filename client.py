import requests 
import os

api_url = "http://127.0.0.1:8000/image_url/"


file_name = "images"

with open(file_name, "r") as file:
    for line in file:
        object_name = line.rstrip()
        #TODO resize images so they are note that big

        try:
            file_size = os.path.getsize(object_name)
        except Exception as e:
            print(f"{e}")
            continue

        response = requests.get(api_url + object_name)
        if response is None:
            print(f"Failed to get upload url for {object_name}")
            continue
        response = response.json()
        
        with open(object_name, 'rb') as f:
            files = {'file': (object_name, f)}
            http_response = requests.post(response['url'], data=response['fields'], files=files)

        # If successful, returns HTTP status code 204
        if http_response.status_code == 204:
            print(f'Succefully uploaded file {object_name}')
