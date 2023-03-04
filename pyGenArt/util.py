import pandas as pd
import requests
import numpy as np
import json
import os
from os.path import join, isdir, isfile
from PIL import Image
import matplotlib.pyplot as plt
from os import listdir

# DEPRECATED: Uploads list of files with known index.
def uploadFiles2Pinata(filelist : str,
                       foldername : str,
                       config):
    files = []
    filenames = []
    for fn in filelist:
        files.append(('file', (join(foldername, fn.split("/")[-1]), open(fn, "rb"))))
        filenames.append(fn.split("/")[-1])

    response = requests.post(
        config.PINATA_BASE_URL + config.endpoint,
        files=files,
        headers=config.headers,
    )
    
    ipfshash = response.json()["IpfsHash"]
    return ipfshash, filenames

# Returns string formatted csv of gdrive chart.
def getGDriveAsCSV(gdrive_id : str):
    response = requests.get('https://docs.google.com/spreadsheet/ccc?key=%s&output=csv'%(gdrive_id))
    assert response.status_code == 200, 'Wrong status code'
    content = response.content.decode('utf-8').split("\r\n")
    content = [i.split(",") for i in content]
    df = pd.DataFrame(content[1:], columns=content[0])
    return df

# Returns number of traits that are different between two NFTs
def getHammingDistance(x, y):
    traits = x.traits_to_diff
    xcode = [x.attrs[key].tid for key in x.attrs if key in traits]
    ycode = [y.attrs[key].tid for key in y.attrs if key in traits]
    
    output = sum([xcode[i]!=ycode[i] for i in range(len(traits))])
    return output

# Given a filename, loads an alpha map.
def loadAlphaClip(filename, th=128):
    talpha = np.array(Image.open(filename))[:, :, -1:]
    # Figure out how to clip
    return talpha > th