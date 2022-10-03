import string
import os
import random
import numpy as np
from dotenv import load_dotenv
load_dotenv()

filenameSize = int(os.getenv('file_name_size'))
saveDir      = os.getenv('save_dir')
publicDir    = os.getenv('public_dir')
modelsDir    = os.getenv('models_dir')


def randomFilename():
    letters = string.ascii_lowercase + string.ascii_uppercase
    while True:
        fileName = ''.join(random.choice(letters) for _ in range(filenameSize))    
        fileName = f'{publicDir}/{saveDir}/{fileName}'
        if not os.path.exists(fileName):
            return fileName

def loadColors():
    names = os.getenv('names_file')
    classes = []
    with  open(names, 'r') as f:
        classes = [i.strip() for i in f.readlines()]
    return np.random.uniform(0, 255, size = (len(classes), 3)), classes

def getModel(modelDir):
    modelPath = os.path.join(modelsDir, modelDir)
    weights   = os.path.join(modelPath, os.listdir(modelPath)[0]) 
    return weights