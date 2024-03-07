import os.path
from typing import Literal

import pandas as pd
import torch

from drugex.logs import logger
from drugex.training.interfaces import TrainingMonitor, Model


class NullMonitor(TrainingMonitor):

    def getSaveModelOption(self) -> Literal['best', 'all', 'improvement']:
        pass

    def saveModel(self, model, identifier=None):
        pass

    def savePerformanceInfo(self, performance_dict, df_smiles=None):
        pass

    def saveProgress(self, model: Model, current_step=None, current_epoch=None, total_steps=None, total_epochs=None, *args, **kwargs):
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

    def __init__(
            self,
            path,
            save_smiles=False,
            save_model_option: Literal['best', 'all', 'improvement'] = 'best',
            reset_directory=False,
            on_model_update=None
    ):
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
            The path and prefix of the files to be created (i.e. /tmp/drugex_rl/experiment_01). This will ensure all files
            are saved to the given directory and have the given prefix.
        save_smiles : bool
            Whether to save the SMILES of the molecules generated in each epoch.
        save_model_option : str
            Determines which models to save during training. Use this parameter with care as saving a large number of
            models is extremely memory-intensive. Possible values:
            - 'all' : Save all models.
            - 'improvement': Save all models that improve upon the previous best model.
            - 'best' (default): Save only the final best model.
            WARNING: Setting this option to 'all' or 'best' can be extremely memory-intensive, Use with caution and 
            ensure you have sufficient memory resources.
        reset_directory : bool
            Whether to reset the directory where the files are to be saved. If `True`, all files
            with the given prefix will be removed from the directory upon creation of the monitor.
        on_model_update : callable
            A callable that will be called after each model update/epoch. The callable will be passed a `Model` subclass
            instance as the only argument.
        """
        
        self.path = path
        self.directory = os.path.dirname(path)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        elif reset_directory:
            for file in os.listdir(self.directory):
                if file.startswith(os.path.basename(path)):
                    logger.warning(f"Removing {file} from {self.directory}")
                    os.remove(os.path.join(self.directory, file))
        self.outLog = open(path + '_fit.log', 'w', encoding='utf-8')
        self.outDF = path + '_fit.tsv'
        self.outSmiles = path + '_smiles.tsv' if save_smiles else None
        self.outSmilesHeaderDone = False
        self.currentState = None
        self.saveModelOption = save_model_option
        self.onModelUpdate = on_model_update

    def saveModel(self, model, identifier=None):
        """ 
        Save the model state.
        """
        self.currentState = model.getModel() 
        suffix = '_' + str(identifier) if identifier else ''
        torch.save(self.currentState, self.path + suffix + '.pkg')

    def saveProgress(self, model: Model, current_step=None, current_epoch=None, total_steps=None, total_epochs=None, loss=None, *args, **kwargs):
        """ 
        Save the current training progress: epoch, step, loss.

        Parameters
        ----------
        model : Model
            The model currently being trained.
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
        if self.onModelUpdate:
            self.onModelUpdate(model)
        
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
        return self.currentState

    def getSaveModelOption(self):
        return self.saveModelOption
