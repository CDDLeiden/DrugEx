import os

from drugex import VERSION
from drugex.logs import logger

def getVocPaths(data_path, voc_files, mol_type):

    voc_paths = []
    for v in voc_files:
        path = data_path + f"{v}_{mol_type}_voc.txt"
        if not os.path.exists(path):
            logger.warning(f'Reading {mol_type}_voc.txt instead of {path}')
            path = data_path + f"{mol_type}_voc.txt"
        if os.path.exists(path):
            voc_paths.append(path)
        else:
            logger.warning(f"No vocabulary files found. Using internal defaults for DrugEx v{VERSION}.")

    return voc_paths

def getDataPaths(data_path, input_prefix, mol_type, unique_frags):

    # If exact data path was given as input, that data is both used for training and testing
    if os.path.exists(data_path + input_prefix):
        train_path = data_path + input_prefix
        test_path = train_path
    # Else if prefix was given, read separate train and test sets
    else:
        train_path = data_path + '_'.join([input_prefix, 'unique' if unique_frags else 'train', mol_type]) + '.txt'
        test_path = data_path + '_'.join([input_prefix, 'test', mol_type]) + '.txt'    
    assert os.path.exists(train_path), f'{train_path} does not exist'
    assert os.path.exists(test_path), f'{test_path} does not exist'          
        
    logger.info(f'Loading training data from {train_path}')
    logger.info(f'Loading validation data from {test_path}')

    return train_path, test_path