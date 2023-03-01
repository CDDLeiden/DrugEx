"""
monitors

Created by: Martin Sicho
On: 02.06.22, 13:59
"""
import os.path
import shutil
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch

from drugex.training.interfaces import TrainingMonitor

class NullMonitor(TrainingMonitor):

    def saveModel(self, model):
        pass

    def savePerformanceInfo(self, performance_dict, df_smiles=None):
        pass

    def saveProgress(self, current_step=None, current_epoch=None, total_steps=None, total_epochs=None, *args, **kwargs):
        pass

    def endStep(self, step, epoch):
        pass

    def close(self):
        pass

    def getModel(self):
        pass

class FileMonitor(TrainingMonitor):
    """
    A simple `TrainingMonitor` implementation with file outputs.

    """

    def __init__(self, path, save_smiles=False, reset_directory=False):
        """
        Initialize the file monitor.

        The monitor will create three/four files:
        - `path`_fit.tsv - a TSV file with the performance data for each epoch
        - `path`_fit.log - a log file with the training progress
        - `path`_smiles.tsv - a TSV file with the SMILES of the molecules generated in each epoch (if `save_smiles` is True)
        - `path`.pkg - a PyTorch package with the model state
        
        Parameters
        ----------
        path : str
            The path and prefix of the files to be created. 
        save_smiles : bool
            Whether to save the SMILES of the molecules generated in each epoch.
        reset_directory : bool
            Whether to reset the directory where the files are to be saved. If True, the directory will be deleted and
            recreated. If False, the files will be appended to the existing directory.
        """
        
        self.path = path
        self.directory = os.path.dirname(path)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        elif reset_directory:
            shutil.rmtree(self.directory)
            os.makedirs(self.directory)
        self.outLog = open(path + '_fit.log', 'w', encoding='utf-8')
        self.outDF = path + '_fit.tsv'
        self.outSmiles = path + '_smiles.tsv' if save_smiles else None
        self.outSmilesHeaderDone = False
        self.bestState = None

    def saveModel(self, model):
        """ 
        Save the model state.
        """
        self.bestState = model.getModel()
        torch.save(self.bestState, self.path + '.pkg')

    def saveProgress(self, current_step=None, current_epoch=None, total_steps=None, total_epochs=None, loss=None, *args, **kwargs):
        """ 
        Save the current training progress: epoch, step, loss.

        Parameters
        ----------
        current_step : int
            The current step.
        current_epoch : int
            The current epoch.
        total_steps : int
            The total number of steps.
        total_epochs : int 
            The total number of epochs.
        loss : float
            The current training loss.
        """
        
        txt = f"Epoch {current_epoch if current_epoch is not None else '--'}/"
        txt += f"{total_epochs if total_epochs is not None else '--'}," 
        txt += f"Step {current_step if current_step is not None else '--'}/"
        txt += f"{total_steps if total_steps is not None else '--'}\n"
        self.outLog.write(txt)
        if loss:
            self.outLog.write(f"Current training loss: {loss:.4f} \n")
        self.outLog.flush()

    def savePerformanceInfo(self, performance_dict, df_smiles=None):
        """ 
        Save the performance data for the current epoch.
        
        Parameters
        ----------
        performance_dict : dict
            A dictionary with the performance data.
        df_smiles : pd.DataFrame
            A DataFrame with the SMILES of the molecules generated in the current epoch.
        """

        df = pd.DataFrame(performance_dict, index=[0])
        self.saveEpochData(df)

        # Save smiles and their indivudual scores if requested
        if self.outSmiles and df_smiles is not None:
            self.saveMolecules(df_smiles)

    def saveEpochData(self, df):
        self.appendTableToFile(df, self.outDF)

    def saveMolecules(self, df):
        if self.outSmiles:
            self.appendTableToFile(df, self.outSmiles)

    @staticmethod
    def appendTableToFile(df, outfile):
        header_written = os.path.isfile(outfile)
        open_mode = 'a' if header_written else 'w'
        df.round(decimals=3).to_csv(
            outfile,
            sep='\t',
            index=False,
            header=not header_written,
            mode=open_mode,
            encoding='utf-8',
            na_rep='NA'
        )

    def endStep(self, step, epoch):
        super().endStep(step, epoch)
        self.outLog.flush()

    def close(self):
        super().close()
        self.outLog.close()
   
    def getModel(self):
        return self.bestState
