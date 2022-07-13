import string
import random
import os
import uvicorn
import shutil
from typing import List
from fastapi import FastAPI, UploadFile
from fastapi.staticfiles import StaticFiles

from fastapi.middleware.cors import CORSMiddleware


from opencv import apiImage, apiVideo

filenameSize = 10
saveDir = 'public'
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=['*']
)

#change the rgb color of the images that should imporve the results
# maybe change for other kind of file systmem if there is time
app.mount("/public", StaticFiles(directory=saveDir), name="public")

def randomFilename():
    letters = string.ascii_lowercase + string.ascii_uppercase
    while True:
        fileName  = ''.join(random.choice(letters) for _ in range(filenameSize))    
        fileName = f'{saveDir}/images/{fileName}'
        if not os.path.exists(fileName):
            return fileName

@app.get('/')
def get():
    return {'meesage': "Hello world"}


@app.post('/image')
async def file(file: UploadFile):
    ext = file.filename.split('.')[-1]
    fileName = randomFilename()
    fileName = f'{fileName}.{ext}'
    with open(f'{fileName}', 'wb') as f:
        shutil.copyfileobj(file.file, f)
    apiImage(fileName)
    return {'filename':fileName}

@app.post('/video')
async def file(file: UploadFile):
    ext = file.filename.split('.')[-1]
    fileName = randomFilename()
    fileName = f'{fileName}.{ext}'
    with open(f'{fileName}', 'wb') as f:
        shutil.copyfileobj(file.file, f)
    videoPath = apiVideo(fileName)
    return {'filename':videoPath}

# process image

@app.post('/uploads')
async def file(files: List[UploadFile]):
    savedFiles = []
    for file in files:
        ext = file.filename.split('.')[-1]
        fileName = randomFilename()
        fileName = f'{fileName}.{ext}'
        savedFiles.append(fileName)
        with open(f'{fileName}', 'wb') as f:
            shutil.copyfileobj(file.file, f)
    return {'filename':savedFiles}


if __name__ == '__main__':
     uvicorn.run("main:app", host='0.0.0.0', port=8000, reload=False, debug=False)