# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

import numpy as np
import pandas as pd

class TabularDataset:
    """
    A class providing facilities for preprocessing and postprocessing of ImageNet validation dataset.
    """

    def __init__(self, file_path, batch_size=8, task='classification'):       
        self.__batch_size = batch_size
        self.__y_true = []
        self.__y_pred = []
        self.__k = 0
        self.task = task
        df = pd.read_csv(file_path)
        self.X = df.iloc[:,:-1].values
        self.y = df.iloc[:,-1].values
        self.available_instances = float('inf')

    def __next__(self):    
        indx = np.random.choice(self.X.shape[0], size=self.__batch_size, replace=True)        
        return self.X[indx].astype(np.float32), self.y[indx]

        
    def reset(self):
        self.__y_true = []
        self.__y_pred = []
        self.__k = 0
        return True    

    def submit_predictions(self, y, y_hat):
        """

        """
        
        self.__y_true += y.tolist()
        self.__y_pred += y_hat.ravel().tolist()
        self.__k += len(y)

    def summarize_accuracy(self):
        """
        A function summarizing the accuracy achieved on the images obtained with get_input_array() calls on which
        predictions done where supplied with submit_predictions() function.
        """
        
        if self.task == 'classification':
            print(f"\nAccuracy based on {self.__k} samples.")
            acc = (np.array(self.__y_true) == np.array(self.__y_pred)).mean()
            print('Acurracy: ', acc)
            return {"accuracy": acc}
        else:
            print(f"\nRMSE based on {self.__k} samples.")            
            rmse = ((np.array(self.__y_true) - np.array(self.__y_pred))**2).mean()**0.5
            print('rmse: ', rmse)
            return {"rmse": rmse}
