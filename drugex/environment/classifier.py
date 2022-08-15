import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import time
import drugex
import inspect
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset


class Base(nn.Module):
    """ This class is the base structure for all of classification/regression DNN models.
    Mainly, it provides the general methods for training, evaluating model and predicting the given data.
    """

    def __init__(self, n_epochs=100, lr=1e-4, batch_size=32):
        super().__init__()
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size=32
        self.dev = drugex.DEFAULT_DEVICE

    def fit(self, train_loader, valid_loader, out):
        """Training the DNN model, similar to the scikit-learn or Keras style.
        In the end, the optimal value of parameters will also be persisted on the hard drive.
        Arguments:
            train_loader (DataLoader): Data loader for training set,
                including m X n target FloatTensor and m X l label FloatTensor
                (m is the No. of sample, n is the No. of features, l is the No. of classes or tasks)
            valid_loader (DataLoader): Data loader for validation set.
                The data structure is as same as loader_train.
            out (str): the file path for the model file (suffix with '.pkg')
                and log file (suffix with '.log').
            epochs(int, optional): The maximum of training epochs (default: 100)
            lr (float, optional): learning rate (default: 1e-4)
        """

        if 'optim' in self.__dict__:
            optimizer = self.optim
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.lr)

        # record the minimum loss value based on the calculation of loss function by the current epoch
        best_loss = np.inf
        # record the epoch when optimal model is saved.
        last_save = 0
        log = open(out + '.log', 'a')
        for epoch in range(self.n_epochs):
            t0 = time.time()
            # decrease learning rate over the epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr * (1 - 1 / self.n_epochs) ** (epoch * 10)
            for i, (Xb, yb) in enumerate(train_loader):
                # Batch of target tenor and label tensor
                Xb, yb = Xb.to(self.dev), yb.to(self.dev)
                optimizer.zero_grad()
                # predicted probability tensor
                y_ = self(Xb, istrain=True)
                # ignore all of the NaN values
                ix = yb == yb
                yb, y_ = yb[ix], y_[ix]

                # loss function calculation based on predicted tensor and label tensor
                loss = self.criterion(y_, yb)
                loss.backward()
                optimizer.step()
            # loss value on validation set based on which optimal model is saved.
            loss_valid = self.evaluate(valid_loader)
            print('[Epoch: %d/%d] %.1fs loss_train: %f loss_valid: %f' % (
                epoch, self.n_epochs, time.time() - t0, loss.item(), loss_valid), file=log)
            if loss_valid < best_loss:
                torch.save(self.state_dict(), out + '.pkg')
                print('[Performance] loss_valid is improved from %f to %f, Save model to %s' %
                      (best_loss, loss_valid, out + '.pkg'), file=log)
                best_loss = loss_valid
                last_save = epoch
            else:
                print('[Performance] loss_valid is not improved.', file=log)
                # early stopping, if the performance on validation is not improved in 100 epochs.
                # The model training will stop in order to save time.
                if epoch - last_save > 100: break
        print('Neural net fitting completed.', file=log)
        log.close()
        self.load_state_dict(torch.load(out + '.pkg'))

    def evaluate(self, loader):
        """Evaluating the performance of the DNN model.
        Arguments:
            loader (torch.util.data.DataLoader): data loader for test set,
                including m X n target FloatTensor and l X n label FloatTensor
                (m is the No. of sample, n is the No. of features, l is the No. of classes or tasks)
        Return:
            loss (float): the average loss value based on the calculation of loss function with given test set.
        """
        loss = 0
        for Xb, yb in loader:
            Xb, yb = Xb.to(self.dev), yb.to(self.dev)
            y_ = self.forward(Xb)
            ix = yb == yb
            yb, y_ = yb[ix], y_[ix]
            loss += self.criterion(y_, yb).item()
        loss = loss / len(loader)
        return loss

    def predict(self, loader):
        """Predicting the probability of each sample in the given dataset.
        Arguments:
            loader (torch.util.data.DataLoader): data loader for test set,
                only including m X n target FloatTensor
                (m is the No. of sample, n is the No. of features)
        Return:
            score (ndarray): probability of each sample in the given dataset,
                it is a m X l FloatTensor (m is the No. of sample, l is the No. of classes or tasks.)
        """
        score = []
        for Xb, _ in loader:
            Xb = Xb.to(self.dev)
            y_ = self.forward(Xb)
            score.append(y_.detach().cpu())
        score = torch.cat(score, dim=0).numpy()
        return score

    @classmethod
    def _get_param_names(cls):
        # function copy from sklearn.base_estimator
        # get the class parameter names
        init_signature = inspect.signature(cls.__init__)
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        return sorted([p.name for p in parameters])
    
    def get_params(self, deep=True):
        # function copy from sklearn.base_estimator
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out
    
    def set_params(self, **params):
        # function copy from sklearn.base_estimator
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                local_valid_params = self._get_param_names()
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self}. "
                    f"Valid parameters are: {local_valid_params!r}."
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def get_dataloader(self, X, y):
        """
            Convert data to tensors and get iterable over dataset with dataloader
            arguments:
            X (numpy 2d array): input dataset
            y (numpy 1d column vector): output data
        """
        tensordataset = TensorDataset(torch.Tensor(X), torch.Tensor(y))
        return DataLoader(tensordataset, batch_size=self.batch_size)
    
class STFullyConnected(Base):
    """Single task DNN classification/regression model. It contains four fully connected layers between which
        are dropout layer for robustness.
    Arguments:
        n_dim (int): the No. of columns (features) for input tensor
        n_class (int): the No. of columns (classes) for output tensor.
        is_reg (bool, optional): Regression model (True) or Classification model (False)
        n_eopchs (int): max number of epochs
        lr (float): neural net learning rate
        neurons_h1 (int): number of neurons in first hidden layer
        neurons_hx (int): number of neurons in other hidden layers
        extra_layer (bool): add third hidden layer
    """

    def __init__(self, n_dim, n_epochs = 100, lr = 1e-4, batch_size=32,
                 is_reg=True, neurons_h1 = 4000, neurons_hx = 1000, extra_layer = False):
        super().__init__(n_epochs = n_epochs, lr = lr, batch_size=batch_size)
        self.n_dim = n_dim
        self.n_class = 1
        self.is_reg = is_reg
        self.neurons_h1 = neurons_h1
        self.neurons_hx = neurons_hx
        self.extra_layer = extra_layer
        self.init_model()

    def init_model(self):
        self.dropout = nn.Dropout(0.25)
        self.fc0 = nn.Linear(self.n_dim, self.neurons_h1)
        self.fc1 = nn.Linear(self.neurons_h1, self.neurons_hx)
        if self.extra_layer:
            self.fc2 = nn.Linear(self.neurons_hx, self.neurons_hx)
        self.fc3 = nn.Linear(self.neurons_hx, self.n_class)
        if self.is_reg:
            # loss function for regression
            self.criterion = nn.MSELoss()
        elif self.n_class == 1:
            # loss function and activation function of output layer for binary classification
            self.criterion = nn.BCELoss()
            self.activation = nn.Sigmoid()
        else:
            # loss function and activation function of output layer for multiple classification
            self.criterion = nn.CrossEntropyLoss()
            self.activation = nn.Softmax()
        self.to(self.dev)

    def set_params(self, **params):
        super().set_params(**params)
        self.init_model()

    def forward(self, X, istrain=False):
        """Invoke the class directly as a function
        Arguments:
            X (FloatTensor): m X n FloatTensor, m is the No. of samples, n is the No. of features.
            istrain (bool, optional): is it invoked during training process (True) or just for prediction (False)
        Return:
            y (FloatTensor): m X l FloatTensor, m is the No. of samples, n is the No. of classes
        """
        y = F.relu(self.fc0(X))
        if istrain:
            y = self.dropout(y)
        y = F.relu(self.fc1(y))
        if self.extra_layer:
            if istrain:
                y = self.dropout(y)
            y = F.relu(self.fc2(y))
        if istrain:
            y = self.dropout(y)
        if self.is_reg:
            y = self.fc3(y)
        else:
            y = self.activation(self.fc3(y))
        return y

# TODO: implement multi-task DNN
# class MTFullyConnected(Base):
#     """Multi-task DNN classification/regression model. It contains four fully connected layers
#     between which are dropout layer for robustness.
#     Arguments:
#         n_dim (int): the No. of columns (features) for input tensor
#         n_task (int): the No. of columns (tasks) for output tensor.
#         is_reg (bool, optional): Regression model (True) or Classification model (False)
#     """

#     def __init__(self, n_dim, n_task, is_reg=False):
#         super(MTFullyConnected, self).__init__()
#         self.n_task = n_task
#         self.dropout = nn.Dropout(0.25)
#         self.fc0 = nn.Linear(n_dim, 4000)
#         self.fc1 = nn.Linear(4000, 2000)
#         self.fc2 = nn.Linear(2000, 1000)
#         self.output = nn.Linear(1000, n_task)
#         self.is_reg = is_reg
#         if is_reg:
#             # loss function for regression
#             self.criterion = nn.MSELoss()
#         else:
#             # loss function and activation function of output layer for multiple classification
#             self.criterion = nn.BCELoss()
#             self.activation = nn.Sigmoid()
#         self.to(self.dev)

#     def forward(self, X, istrain=False):
#         """Invoke the class directly as a function
#         Arguments:
#             X (FloatTensor): m X n FloatTensor, m is the No. of samples, n is the No. of features.
#             istrain (bool, optional): is it invoked during training process (True)
#                 or just for prediction (False)
#         Return:
#             y (FloatTensor): m X l FloatTensor, m is the No. of samples, n is the No. of tasks
#         """
#         y = F.relu(self.fc0(X))
#         if istrain:
#             y = self.dropout(y)
#         y = F.relu(self.fc1(y))
#         if istrain:
#             y = self.dropout(y)
#         y = F.relu(self.fc2(y))
#         if istrain:
#             y = self.dropout(y)
#         if self.is_reg:
#             y = self.output(y)
#         else:
#             y = self.activation(self.output(y))
#         return y

# TODO: Is this model useful?
# class Discriminator(Base):
#     """A CNN for text classification
#     architecture: Embedding >> Convolution >> Max-pooling >> Softmax
#     """

#     def __init__(self, vocab_size, emb_dim, filter_sizes, num_filters, dropout=0.25):
#         super(Discriminator, self).__init__()
#         self.emb = nn.Embedding(vocab_size, emb_dim)
#         self.convs = nn.ModuleList([
#             nn.Conv2d(1, n, (f, emb_dim)) for (n, f) in zip(num_filters, filter_sizes)
#         ])
#         self.highway = nn.Linear(sum(num_filters), sum(num_filters))
#         self.dropout = nn.Dropout(p=dropout)
#         self.lin = nn.Linear(sum(num_filters), 1)
#         self.sigmoid = nn.Sigmoid()
#         self.init_parameters()

#         self.to(self.dev)
#         self.optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()))

#     def forward(self, x):
#         """
#         Args:
#             x: (batch_size * seq_len)
#         """
#         emb = self.emb(x).unsqueeze(1)  # batch_size * 1 * seq_len * emb_dim
#         convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]  # [batch_size * num_filter * length]
#         pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]  # [batch_size * num_filter]
#         pred = torch.cat(pools, 1)  # batch_size * num_filters_sum
#         highway = self.highway(pred)
#         pred = highway.sigmoid() * F.relu(highway) + (1. - highway.sigmoid()) * pred
#         pred = self.sigmoid(self.lin(self.dropout(pred)))
#         return pred

#     def init_parameters(self):
#         for param in self.parameters():
#             param.data.uniform_(-0.05, 0.05)