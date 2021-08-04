import numpy as np

class EarlyStop:
    """Early stops the training if validation loss doesn't improve anymore."""

    def __init__(self, stop_criteria, patience=4, verbose=False, delta0=0.01, delta1=0.01):
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
        self.stop = False
        self.delta0 = delta0
        self.delta1 = delta1
        self.loss_mean_vector = []
        self.stop_criteria = stop_criteria

    def __call__(self, val_loss):

        if self.stop_criteria == 0:
            self.early_stop0(val_loss)
        elif self.stop_criteria == 1:
            self.early_stop1(val_loss)

    def early_stop0(self, val_loss):
        score = val_loss
        self.loss_mean_vector.append(val_loss)

        if self.best_score is None:
            self.best_score = score
        elif abs(score - self.best_score) < self.delta0:
            self.best_score = score
            self.counter += 1

            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience} ')

            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_score = score
            self.counter = 0

    def early_stop1(self, val_loss):
        self.loss_mean_vector.append(val_loss)

        if abs(val_loss) < self.delta1:
            self.stop = True



