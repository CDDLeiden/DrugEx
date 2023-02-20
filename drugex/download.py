# -*- coding: utf-8 -*-


import os
import json
import argparse

from drugex.logs import logger
from drugex.logs.utils import enable_file_logger, commit_hash
from drugex.utils.download import download_file
from qsprpred.data.sources.papyrus import Papyrus


def DownloadArgParser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-o', '--out_dir', type=str, default='data',
                        help="Base directory in which to download the tutorial files, should contain the 'tutorial' folder")
    parser.add_argument('-p', '--progress', action='store_true',
                        help="If on, progress of the download is shown")
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-r', '--reload', action='store_true', default=False, help="If on, existing files are re-downloaded.")
    parser.add_argument('-ng', '--no_git', action='store_true', help="If on, git hash is not retrieved")

    args = parser.parse_args()
    return args


def DownloadTutorial(args):
    # Link to DrugEx v2 pretrained model (Papyrus 05.5)
    link_pretrained_model1 = "https://zenodo.org/record/7378923/files/DrugEx_v2_PT_Papyrus05.5.zip?download=1"
    # Link to DrugEx v3 pretrained model (graph-based; Papyrus 05.5)
    link_pretrained_model2 = "https://zenodo.org/record/7085421/files/DrugEx_PT_Papyrus05.5.zip?download=1"
    # Link to QSAR example model
    link_qsar_model = "https://zenodo.org/record/7650233/files/qspr.zip?download=1"

    # Download model files
    pretrained_models_path_rnn = os.path.join(args.out_dir, 'models', 'pretrained', 'smiles-rnn')
    outpath = os.path.join(pretrained_models_path_rnn, 'PT_model1.zip')
    if args.reload or not os.path.exists(pretrained_models_path_rnn):
        download_file(link_pretrained_model1,
                  outpath,
                  os.path.join(pretrained_models_path_rnn, 'Papyrus05.5_smiles_rnn_PT'))

    pretrained_models_path_graph = os.path.join(args.out_dir, 'models', 'pretrained', 'graph-trans')
    outpath = os.path.join(pretrained_models_path_graph, 'PT_model2.zip')
    if args.reload or not os.path.exists(pretrained_models_path_graph):
        download_file(link_pretrained_model2,
                  outpath,
                  os.path.join(pretrained_models_path_graph, 'Papyrus05.5_graph_trans_PT'))

    pretrained_models_path_qsar = os.path.join(args.out_dir, 'models', 'qsar')
    outpath = os.path.join(pretrained_models_path_qsar, 'qspr.zip')
    if args.reload or not os.path.exists(pretrained_models_path_qsar):
        download_file(link_qsar_model,
                  outpath,
                  pretrained_models_path_qsar)

    # Download data files
    logger.info("Downloading data files from Papyrus database.")
    acc_keys = ["P29274"]  # Adenosine receptor A2A (https://www.uniprot.org/uniprotkb/P29274/entry)
    dataset_name = "A2AR_LIGANDS"  # name of the file to be generated
    quality = "high"  # choose minimum quality from {"high", "medium", "low"}
    papyrus_version = '05.6'  # Papyrus database version

    papyrus = Papyrus(
        data_dir=os.path.join(args.out_dir, 'datasets', '.Papyrus'),
        stereo=False,
        version=papyrus_version
    )

    datasets_dir = os.path.join(args.out_dir, 'datasets')
    os.makedirs(datasets_dir, exist_ok=True)
    dataset = papyrus.getData(
        acc_keys,
        quality,
        output_dir=datasets_dir,
        name=dataset_name,
        use_existing=True
    )

    print(f"Tutorial data for accession keys '{acc_keys}' was loaded. Molecules in total: {len(dataset.getDF())}")


if __name__ == '__main__':
    args = DownloadArgParser()

    backup_msg = None
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    logSettings = enable_file_logger(
        out_dir,
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
    with open(os.path.join(out_dir, 'download.json'), 'w') as f:
        json.dump(vars(args), f)

    DownloadTutorial(args)
