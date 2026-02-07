import torch
import numpy as np

class EarlyStopping:
    """
    PURPOSE: 
    Monitors validation energy gap and stops training if no improvement is found after a specified 'patience' period.
    """

    def __init__(self,patience=5, min_delta=0,path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False


    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter+=1
            print(f"Early Stopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
    
    def save_checkpoint(self, model):
        "Saves the model when validation loss decreases."
        torch.save(model.state_dict(), self.path)
        print(f"-> Validation energy improved. Saving model artifact...")