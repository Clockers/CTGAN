import numpy as np

class EarlyStop:
    """Early stops the training if validation loss doesn't improve anymore."""

    def __init__(self, patience=4, verbose=False, delta=0.01):
        """
        Args:
            patience (int): Consecutive checks that must be passed.
                            Default: 4, means 5 values with a delta difference between them
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float):  Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0.01
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.loss_mean_vector = []

    def __call__(self, val_loss):

        score = val_loss
        self.loss_mean_vector.append(val_loss)

        if self.best_score is None:
            self.best_score = score
        elif abs(score - self.best_score) < self.delta:
            self.best_score = score
            self.counter += 1

            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience} ')

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
