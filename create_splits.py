import argparse
import glob
import os
import random

import numpy as np

from utils import get_module_logger


def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /mnt/data
    """
    # TODO: Implement function
    listoffiles = [filename for filename in glob.glob(f'{data_dir}/*.tfrecord')]
    
    np.random.shuffle(listoffiles)
    
    # splitting the files
    training_files, validation_files, testing_files = np.split(listoffiles, [int(.75*len(listoffiles)), int(.9*len(listoffiles))])
    
    # create dirs and move data files into them
    training = os.path.join(data_dir, 'train')
    
    # check if directories exist and then move the data files in directories
    try:
        if os.path.exists(training):
            os.makedirs(training)
    except:
        os.makedirs(training,exist_ok=True)
    
    for file in training_files:
        shutil.move(file, training)
    
    validating = os.path.join(data_dir, 'val')
    
    try:
        if os.path.exists(validating):
            os.makedirs(validating)
    except:
        os.makedirs(validating,exist_ok=True)
    
    for file in validating_files:
        shutil.move(file, validating)
    
    testing = os.path.join(data_dir, 'test')
    os.makedirs(testing, exist_ok=True)
    
    for file in testing_files:
        shutil.move(file, testing) 
    

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Splitting data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)