# -*- coding: utf-8 -*-


import os
import json
import time
import argparse

from drugex.logs.utils import enable_file_logger, commit_hash, backUpFilesInFolder, generate_backup_runID
from drugex.utils.download import download_file


def DownloadArgParser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-o', '--out_dir', type=str, default='.',
                        help="Base directory in which to download the tutorial files, should contain the 'tutorial' folder")
    parser.add_argument('-p', '--progress', action='store_true',
                        help="If on, progress of the download is shown")
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-ng', '--no_git', action='store_true', help="If on, git hash is not retrieved")

    args = parser.parse_args()
    return args


def DownloadTutorial(args):
    # Link to ZIP of tutorial data
    link_tutorial_zip = "https://drive.google.com/u/0/uc?id=1lYOmQBnAawnDR2Kwcy8yVARQTVzYDelw&export=download&confirm=t"
    # Link to DrugEx v2 pretrained model (Papyrus 05.5)
    link_pretrained_model1 = "https://zenodo.org/record/7378923/files/DrugEx_v2_PT_Papyrus05.5.zip?download=1"
    # Link to DrugEx v3 pretrained model (graph-based; Papyrus 05.5)
    link_pretrained_model2 = "https://zenodo.org/record/7085421/files/DrugEx_PT_Papyrus05.5.zip?download=1"

    # Download files
    download_file(link_tutorial_zip,
                  os.path.join(args.out_dir, 'tutorial', 'tutorial_data.tar.gz'))
    download_file(link_pretrained_model1,
                  os.path.join(args.out_dir, 'tutorial', 'jupyter', 'models', 'pretrained', 'PT_model1.zip'),
                  os.path.join(args.out_dir, 'tutorial', 'jupyter', 'models', 'pretrained', 'smiles', 'Papyrus05.5_smiles_rnn_PT'))
    download_file(link_pretrained_model2,
                  os.path.join(args.out_dir, 'tutorial', 'jupyter', 'models', 'pretrained', 'PT_model2.zip'),
                  os.path.join(args.out_dir, 'tutorial', 'jupyter', 'models', 'pretrained', 'graph', 'Papyrus05.5_graph_trans_PT'))


if __name__ == '__main__':
    args = DownloadArgParser()

    base_dir = os.path.join(args.out_dir, 'tutorial')
    if not os.path.isdir(base_dir):
        print('ERROR: The current folder does not contain the required \'tutorial\' folder, download aborted.')
    else:
        out_dir = os.path.join(base_dir, 'tutorial_data')
        backup_msg = None
        if os.path.exists(out_dir):
            backup_id = generate_backup_runID(out_dir)
            backup_msg = backUpFilesInFolder(out_dir, backup_id, ('data',), ('log', 'pkg', 'vocab', 'txt', 'tsv', 'md'))

        logSettings = enable_file_logger(
            base_dir,
            'download.log',
            args.debug,
            __name__,
            commit_hash(os.path.dirname(os.path.realpath(__file__))) if not args.no_git else None,
            vars(args)
        )
        log = logSettings.log
        if backup_msg is not None:
            log.info(backup_msg)

        # Create json log file with used commandline arguments
        if args.debug:
            print(json.dumps(vars(args), sort_keys=False, indent=2))
        with open(os.path.join(base_dir, 'download.json'), 'w') as f:
            json.dump(vars(args), f)

        DownloadTutorial(args)
