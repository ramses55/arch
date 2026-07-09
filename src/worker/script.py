from fastapi import FastAPI, Response, Request
from utils import settings
from utils import work
from mysql.connector import pooling
import uvicorn
import os
import traceback
import time
import json


app = FastAPI()


@app.post("/")
async def basic(request: Request):

    event = await request.json()
    print("EVENT:")
    print(event)
    messages = event.get('messages', [])
    for message in messages:
        body = message['details']['message']['body']
        body = json.loads(body)
        path = body["path"]
        filename = body["filename"]
        try:
            time.sleep(1)
            print(f"Started work on  {filename}")
            key=work(path, filename, pool)
            print(f"work returned: {key}")
            print(f"Ended work on  {filename}")
            return Response(status_code=200)

        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()
            return Response(status_code=500)





pool = None


@app.on_event("startup")
async def startup():
    global pool

    pool = pooling.MySQLConnectionPool(
        pool_name="pool1",
        host=settings.mysql_host,
        port=3306,
        user=settings.mysql_user,
        password=settings.mysql_pass,
        database=settings.mysql_db,
        connection_timeout=5,
        pool_size=5,
    )



#@app.on_event("shutdown")
#async def shutdown():
#    pool.close()

if __name__ == "__main__":
    uvicorn.run(app, port=settings.PORT, host="0.0.0.0")

