from drugex.training.scorers.predictors import Predictor
from rdkit import Chem
from sklearn.preprocessing import MinMaxScaler as Scaler
from sklearn.model_selection import StratifiedKFold, KFold
from drugex.logs import logger

class QSARDataset:
    """
        This class is used to prepare the dataset for QSAR model training. 
        It splits the data in train and test set, as well as creating cross-validation folds.
        Optionally low quality data is filtered out.
        For classification the dataset samples are labelled as active/inactive.
        ...

        Attributes
        ----------
        input_df (pd dataframe)   : dataset
        target (str)              : target identifier (accession in papyrus dataset)
        reg (bool)                : if true, dataset for regression, if false dataset for classification
        timesplit (int), optional : Year to split test set on
        test_size (int or float), optional: Used when timesplit is None
                                            If float, should be between 0.0 and 1.0 and is proportion of dataset to
                                            include in test split. If int, represents absolute number of test samples.
        th (float)                : threshold for activity if classficiation model, ignored otherwise
        keep_low_quality (bool)   : if true low quality data is included in the dataset
        n_folds(int)              : number of folds for crossvalidation

        targetcol (str)           : name of column in dataframe for the target identifier
        smilescol (str)           : name of column in dataframe for smiles
        valuecol (str)            : name of column in dataframe for pX values
        qualitycol (str)          : name of column in dataframe for quality of measurements ("Low, Medium, High"),
                                    has no effect if keep_low_quality is True
        timecol (str)             : name of column in dataframe for timesplit has no effect if timesplit is None

        X (np.ndarray)            : m x n feature matrix for cross validation, where m is the number of samples
                                    and n is the number of features.
        y (np.ndarray)            : m-d label array for cross validation, where m is the number of samples and
                                    equals to row of X.
        X_ind (np.ndarray)        : m x n Feature matrix for independent set, where m is the number of samples
                                    and n is the number of features.
        y_ind (np.ndarray)        : m-l label array for independent set, where m is the number of samples and
                                    equals to row of X_ind, and l is the number of types.
        folds (generator)         :

        Methods
        -------
        split_dataset : A train and test split is made.
        create_folds: folds is an generator and needs to be reset after cross validation or hyperparameter optimization
        data_standardization: Performs standardization by centering and scaling
    """
    def __init__(self, input_df, target, reg=True, timesplit=None, test_size=0.1, th=6.5, keep_low_quality=False, n_folds=5,
                 targetcol = 'accession', smilescol = 'SMILES', valuecol = 'pchembl_value_Mean', qualitycol = 'Quality',
                 timecol = 'Year'):
        self.input_df = input_df
        self.target = target
        self.reg = reg
        self.timesplit = timesplit
        self.test_size = test_size
        self.th = th
        self.keep_low_quality = keep_low_quality
        self.n_folds=5

        self.targetcol = targetcol
        self.smilescol = smilescol
        self.valuecol = valuecol
        self.qualitycol = qualitycol
        self.timecol = timecol

        self.X = None
        self.y = None
        self.X_ind = None
        self.y_ind = None
        self.folds = None

    def split_dataset(self):
        """
        Splits the dataset in a train and temporal test set.
        Calculates the predictors for the QSAR models.
        """
        # read in the dataset
        df = self.input_df.dropna(subset=[self.smilescol]) # drops if smiles is missing
        df = df[df[self.targetcol] == self.target].set_index([self.smilescol])

        # filter out low quality data if desired
        df = df if self.keep_low_quality else df[df[self.qualitycol] != 'Low']

        # Get indexes of samples test set based on temporal split
        if self.timesplit:
            year = df[[self.timecol]].groupby([self.smilescol]).max().dropna()
            test_idx = year[year[self.timecol] > self.timesplit].index

        # keep only pchembl values and make binary for classification
        df = df[self.valuecol]

        if not self.reg:
            df = (df > self.th).astype(float)

        # get test and train (data) set with set temporal split or random split
        df = df.sample(len(df)) 
        if self.timesplit:
            test_ix = set(df.index).intersection(test_idx)
            test = df.loc[list(test_ix)].dropna()
        else:
            test = df.sample(int(round(len(df)*self.test_size))) if type(self.test_size) == float else df.sample(self.test_size)
        data = df.drop(test.index)

        # calculate ecfp and physiochemical properties as input for the predictors
        self.X_ind = Predictor.calculateDescriptors([Chem.MolFromSmiles(mol) for mol in test.index])
        self.X = Predictor.calculateDescriptors([Chem.MolFromSmiles(mol) for mol in data.index])

        self.y_ind = test.values
        self.y = data.values

        # Create folds for crossvalidation
        self.create_folds()

        #Write information about the trainingset to the logger
        logger.info('Train and test set created for %s %s:' % (self.target, 'REG' if self.reg else 'CLS'))
        logger.info('    Total: train: %s test: %s' % (len(data), len(test)))
        if self.reg:
            logger.info('    Total: active: %s not active: %s' % (sum(df >= self.th), sum(df < self.th)))
            logger.info('    In train: active: %s not active: %s' % (sum(data >= self.th), sum(data < self.th)))
            logger.info('    In test:  active: %s not active: %s\n' % (sum(test >= self.th), sum(test < self.th)))
        else:
            logger.info('    Total: active: %s not active: %s' % (df.sum().astype(int), (len(df)-df.sum()).astype(int)))
            logger.info('    In train: active: %s not active: %s' % (data.sum().astype(int), (len(data)-data.sum()).astype(int)))
            logger.info('    In test:  active: %s not active: %s\n' % (test.sum().astype(int), (len(test)-test.sum()).astype(int)))
    
    def create_folds(self):
        """
            Create folds for crossvalidation
        """
        if self.reg:
            self.folds = KFold(self.n_folds).split(self.X)
        else:
            self.folds = StratifiedKFold(self.n_folds).split(self.X, self.y)
        logger.debug("Folds created for crossvalidation")
        
    @staticmethod
    def data_standardization(data_x, test_x):
        """
        Perform standardization by centering and scaling

        Arguments:
                    data_x (list): descriptors of data set
                    test_x (list): descriptors of test set
        
        Returns:
                    data_x (list): descriptors of data set standardized
                    test_x (list): descriptors of test set standardized
        """
        scaler = Scaler(); scaler.fit(data_x)
        test_x = scaler.transform(test_x)
        data_x = scaler.transform(data_x)
        logger.debug("Data standardized")
        return data_x, test_x