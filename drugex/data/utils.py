import os

from drugex import VERSION
from drugex.logs import logger

def getVocPaths(data_path, voc_files, mol_type):
    """ 
    Get paths to vocabulary files. If none are found, use internal defaults.
    
    Parameters
    ----------
    data_path : str
        Path to data directory.
    voc_files : list
        List of vocabulary file names.
        
    Returns
    -------
    list
        List of paths to vocabulary files.
    """

    voc_paths = []
    for voc_file in voc_files:
        path = f'{data_path}/{voc_file}'
        if os.path.exists(path):
            voc_paths.append(path)
        elif os.path.exists(path + f'_{mol_type}.txt.vocab'):
            voc_paths.append(path + f'_{mol_type}.txt.vocab')
        else:
            logger.warning(f'Could not find vocabulary file {voc_file} in {data_path}.')
            
    if len(voc_paths) == 0 :
        logger.warning(f'No vocabulary files found. Using internal defaults for DrugEx v{VERSION}.')

    return voc_paths

def getDataPaths(data_path, input_prefix, mol_type, unique_frags):

    """ 
    Get paths to training and test data files.
    
    Parameters
    ----------
    data_path : str
        Path to data directory.
    input_prefix : str
        Prefix of data files. If a file with the exact name exists, it is used for both training and testing.
    mol_type : str
        Type of molecules in data files. Either 'smiles' or 'graph'.
    unique_frags : bool
        Whether to use unique fragments or not.
    
    Returns
    -------
    Tuple[str, str]
        Paths to training and test data files.
    """

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