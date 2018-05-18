
from keras.callbacks import Callback
import matplotlib.pyplot as plt
import keras.backend as K
import numpy as np
from scipy.signal import savgol_filter
import warnings
import math


class LRFinder(Callback):
    '''
    A simple callback for finding the optimal learning rate range for your model + dataset.

    # Usage
        ```python
            lr_finder = LRFinder(min_lr=1e-5,
                                 max_lr=1e-2,
                                 steps_per_epoch=np.ceil(data_size/batch_size),
                                 epochs=5)
            model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[lr_finder])

            lr_finder.plot_loss()
        ```

    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(data_size/batch_size)`.
        epochs: Number of epochs to run experiment. Usually between 1 and 5 is sufficient.

    # References
        Original paper: https://arxiv.org/abs/1506.01186
    '''

    def __init__(self, min_lr=1e-5, max_lr=1e0, steps_per_epoch=None, epochs=None, early_stop=True, linear=False):
        super().__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_iterations = steps_per_epoch * epochs
        self.iteration = 0
        self.history = {}
        self.linear = linear
        self.best_loss = 1e9   # Initialise to some large number
        self.best_acc = 0    # initialise to zero
        self.early_stop = early_stop

        if self.linear:
            self.lr_mult = self.max_lr / self.min_lr / self.total_iterations
        else:
            self.lr_mult = (self.max_lr / self.min_lr) ** (1 / self.total_iterations)


    def clr(self):
        '''Calculate the learning rate.'''
        mult = self.lr_mult*self.iteration if self.linear else self.lr_mult**self.iteration

        return self.min_lr * mult


    def on_train_begin(self, logs=None):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.min_lr)


    def on_batch_end(self, epoch, logs=None):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.iteration += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.iteration)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())

        # Stop LR Range Test if loss increases 2x more than best or accuracy is half of best accuracy
        if self.early_stop == True:
            loss = logs.get('loss')
            acc = logs.get('acc')

            if loss is None:
                warnings.warn("Early stopping requires %s available!" % 'loss', RuntimeWarning)

            if loss < self.best_loss: self.best_loss = loss
            if acc > self.best_acc: self.best_acc = acc

            if math.isnan(loss) or loss > (self.best_loss*3) or acc < (self.best_acc/3):
                print('\nBatch %05d: LR range test -- early stopping' % epoch)
                self.model.stop_training = True


    def plot_lr(self):
        '''Helper function to quickly inspect the learning rate schedule.'''
        plt.figure(figsize=(18,4))

        plt.subplot(1, 3, 1)
        plt.plot(self.history['iterations'], self.history['lr'])
        plt.yscale('linear')
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Learning rate', fontsize=14)

        plt.subplot(1, 3, 2)
        plt.plot(self.history['lr'], self.history['loss'])
        plt.xscale('log')
        plt.xlabel('Learning rate', fontsize=14)
        plt.ylabel('Loss', fontsize=14)

        plt.subplot(1, 3, 3)
        plt.plot(self.history['lr'], self.history['acc'])
        plt.xscale('log')
        plt.xlabel('Learning rate', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)

        plt.show()


    def plot(self, monitor='loss', skip_start=10, skip_end=5):
        '''Helper function to quickly observe the learning rate experiment results.'''
        result = savgol_filter(self.history[monitor], 51, 2) # window size 51, polynomial order 2
        # result = smooth_curve(self.history[monitor], 0.98)

        dresult = np.array(result[1:]) - np.array(result[:-1])
        dresult = savgol_filter(dresult, 51, 2) # window size 51, polynomial order 2
        # dresult = smooth_curve(dresult, 0.98)

        fig, ax1 = plt.subplots(figsize=(8,6))
        ax2 = ax1.twinx()

        ax1.plot(self.history['lr'][skip_start:-skip_end], result[skip_start:-skip_end], 'black')
        ax1.set_ylabel(monitor, color='black', fontsize=16)
        ax1.tick_params('y', colors='black')

        ax2.plot(self.history['lr'][skip_start+1:-skip_end], dresult[skip_start:-skip_end], 'blue', linestyle='--')
        ax2.set_ylabel('d' + monitor, color='blue', fontsize=16)
        ax2.tick_params('y', colors='blue')

        ax1.set_xlabel('Learning Rate', fontsize=16)
        ax1.set_xscale('log')


def smooth_curve(vals, beta):
    avg_val = 0
    smoothed = []
    for (i,v) in enumerate(vals):
        avg_val = beta * avg_val + (1-beta) * v
        smoothed.append(avg_val/(1-beta**(i+1)))
    return smoothed
