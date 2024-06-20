"""
This file contains some functions and classes which can be useful in very diverse projects.
"""
import os
import sys
import torch
import logging
import traceback
from os.path import join
import random
import numpy as np
import sys
import requests
from bs4 import BeautifulSoup


def submit_poses(method, path, dataset='aachenv11'):
    session = open('ign_secret.txt', 'r').readline().strip()
    resp = requests.get("https://www.visuallocalization.net/submission/",
            headers={"Cookie": f"sessionid={session}"})

    bs = BeautifulSoup(resp.content, features="html.parser")
    csrf = bs.select('form > input[name="csrfmiddlewaretoken"]')[0].attrs['value']

    url = "https://www.visuallocalization.net/submission/"

    headers = {
            "Cookie": f"csrftoken={resp.cookies['csrftoken']}; sessionid={session}",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36",
    }

    data = {
            "csrfmiddlewaretoken": csrf,
            "method_name": method, #"test",
            "publication_url": "",
            "code_url": "",
            "info_field": "",
            "dataset": dataset, 
            "workshop_submission": "default",
    }

    files = {
            "result_file": open(path, "r"),
    }
    resp = requests.post(url, files=files, data=data, headers=headers)
    print(resp.status_code)


def setup_logging(output_folder, console="debug",
                  info_filename="info.log", debug_filename="debug.log"):
    """Set up logging files and console output.
    Creates one file for INFO logs and one for DEBUG logs.
    Args:
        output_folder (str): creates the folder where to save the files.
        debug (str):
            if == "debug" prints on console debug messages and higher
            if == "info"  prints on console info messages and higher
            if == None does not use console (useful when a logger has already been set)
        info_filename (str): the name of the info file. if None, don't create info file
        debug_filename (str): the name of the debug file. if None, don't create debug file
    """
    if os.path.exists(output_folder):
        raise FileExistsError(f"{output_folder} already exists!")
    os.makedirs(output_folder, exist_ok=True)
    # logging.Logger.manager.loggerDict.keys() to check which loggers are in use
    logging.getLogger('PIL').setLevel(logging.INFO)  # turn off logging tag for some images
    logging.getLogger('matplotlib.font_manager').disabled = True
    
    base_formatter = logging.Formatter('%(asctime)s   %(message)s', "%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    
    if info_filename != None:
        info_file_handler = logging.FileHandler(join(output_folder, info_filename))
        info_file_handler.setLevel(logging.INFO)
        info_file_handler.setFormatter(base_formatter)
        logger.addHandler(info_file_handler)
    
    if debug_filename != None:
        debug_file_handler = logging.FileHandler(join(output_folder, debug_filename))
        debug_file_handler.setLevel(logging.DEBUG)
        debug_file_handler.setFormatter(base_formatter)
        logger.addHandler(debug_file_handler)
    
    if console != None:
        console_handler = logging.StreamHandler()
        if console == "debug": console_handler.setLevel(logging.DEBUG)
        if console == "info":  console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(base_formatter)
        logger.addHandler(console_handler)
    
    def exception_handler(type_, value, tb):
        logger.info("\n" + "".join(traceback.format_exception(type, value, tb)))
    sys.excepthook = exception_handler


def make_deterministic(seed=0):
    """Make results deterministic. If seed == -1, do not make deterministic.
        Running your script in a deterministic way might slow it down.
        Note that for some packages (eg: sklearn's PCA) this function is not enough.
    """
    seed = int(seed)
    if seed == -1:
        return
    print(f'setting seed {seed}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

