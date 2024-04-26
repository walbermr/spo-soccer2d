import os
import sys
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
import torch
import numpy as np
from numpy.linalg import norm
import pandas as pd
import torch.nn as nn
import pytorch_lightning as pl

from torchmetrics.functional import accuracy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from torch.utils.data import Dataset, DataLoader

from typing import Any, List, Optional, Union
import multiprocessing

from tqdm import tqdm

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

class TabularDataset(Dataset):
    '''
    Defines the class to handle tabular data.

    Parameters
    ----------
    X : np.array
        Samples for the dataset, in np.array format.
    y : np.array
        Classes, or labels, for the dataset samples.
    wm : np.array
        Classes, or labels, for the dataset samples

    Attributes
    ----------
    X : np.array
        Samples for the dataset, in np.array format.
    y : np.array
        Classes, or labels, for the dataset samples.
    wm : np.array
        Classes, or labels, for the dataset samples
    '''
    def __init__(self, X: np.array, y: np.array, one_hot_encoder: OneHotEncoder) -> None:
        super().__init__()
        
        self.X, self.y = X, y
        self.enc = one_hot_encoder
        

    def __getitem__(self, index:int) -> dict:
        '''
        Function that returs the 'index' sample from the dataset. Required when using pytorch's DataLoader. Note that this function
        does not require a dictionary as return type, as this output will be handled by the programmer in the model's training function.

        Parameters
        ----------
        index : int
            Index to get sample

        Returns
        -------
        dictionary containing 'X' and 'y' as keys.
        '''
        encoded_row = self.enc.transform(self.y[index][np.newaxis]).A[0] #pythonic idiomatic
        
        return {
                'X': self.X[index],
                'y': encoded_row,
                'label': np.argmax(encoded_row) + 1,
            }

    def __len__(self) -> int:
        '''
        Function that returs the size of the dataset. Required when using pytorch's DataLoader.

        Returns
        -------
        integer the dataset's size.
        '''
        return self.X.shape[0]


class DataModule(pl.LightningDataModule):
    '''
    Defines the DataModule. This pytorch lightning class contains all the required handling for the datasets: train, test, and validation.
    This class is used to ease the implementation and handling of all datasets by the training module.

    Attributes
    ----------
    train_dataset : TabularDataset
        A Dataset derived class to handle tabluar data during training.
    test_dataset : TabularDataset
        A Dataset derived class to handle tabluar data during testing.
    val_dataset : TabularDataset
        A Dataset derived class to handle tabluar data during validation.
    '''
    def __init__(self, num_workers=4) -> None:
        super().__init__()
        basedir_csv = './db/cyrus-dataset-agent2d/'
        list_dfs = []
        
        for dirpath, dirnames, filenames in os.walk(basedir_csv):
            for file in filenames:
                if '.csv' not in file:
                    continue
                df = pd.read_csv(os.path.join(dirpath, file), index_col=False)
                # print(dirpath + file)
                list_dfs.append(df)
        
        full_df = pd.concat(list_dfs)
        
        a = ["out", "pass"]
        feats = []
        # for column in full_df.columns:
        #     if "out" in column or "pass" in column:
        #         feats.append(column)

        feats = list(set(feats) - set(["out_unum"]))
        full_df = full_df.drop(columns=feats+["Unnamed: 781"])
        full_df = full_df.dropna()
       
        train_split, test_split = train_test_split(full_df, test_size=0.3, random_state=199)
        test_split, val_split = train_test_split(test_split, test_size=0.5, random_state=199)
        # drop any?

        # get y
        y_columns = ['out_unum']
        train_split_y = train_split[y_columns]
        train_split = train_split.drop(columns=y_columns)
        
        test_split_y = test_split[y_columns]
        test_split = test_split.drop(columns=y_columns)

        val_split_y = val_split[y_columns]
        val_split = val_split.drop(columns=y_columns)

        fake_y = np.arange(start=1, stop=12, step=1).reshape((11, 1))
        enc = OneHotEncoder(handle_unknown='ignore').fit(fake_y)
        
        self.train_dataset = TabularDataset(train_split.values.astype(np.float32), train_split_y.values.astype(np.float32), enc)
        self.test_dataset = TabularDataset(test_split.values.astype(np.float32), test_split_y.values.astype(np.float32), enc)
        self.val_dataset = TabularDataset(val_split.values.astype(np.float32), val_split_y.values.astype(np.float32), enc)
        self.num_workers = num_workers

    def train_dataloader(self, batch_size=64) -> DataLoader:
        '''
        Defines the train_dataloader get method, used by the pl.Trainer.fit function.

        Returns
        -------
        DataLoader to handle train_dataset.
        '''
        return DataLoader(self.train_dataset, batch_size=batch_size, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self, batch_size=64) -> DataLoader:
        '''
        Defines the val_dataloader get method, used by the pl.Trainer.fit function.

        Returns
        -------
        DataLoader to handle val_dataset.
        '''
        return DataLoader(self.val_dataset, batch_size=batch_size, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self, batch_size=64) -> DataLoader:
        '''
        Defines the test_dataloader get method, used by the pl.Trainer.fit function.

        Returns
        -------
        DataLoader to handle test_dataset.
        '''
        return DataLoader(self.test_dataset, batch_size=batch_size, num_workers=self.num_workers, persistent_workers=True)

# Cyrus Model transcribed from 2DAgent-DataExtractor
class CyrusModel(nn.Module): 
    def __init__(self, output_size=11):
        super().__init__()
        
        self.backbone = nn.Sequential(
            nn.Linear(780, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(inplace=True),
            nn.Linear(32, output_size),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class CyrusPassClassifier(pl.LightningModule):
    def __init__(self, *args, **kwargs: Any) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        self.lr = 2e-4
        self.model = CyrusModel()
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: dict, batch_idx: int):
        (
            x_row, 
            y_encoded,
            _y_label,
        ) = (
            batch["X"], 
            batch["y"],
            batch["label"],
        )
        
        y_hat = self.model(x_row)
        loss = torch.nn.functional.cross_entropy(y_hat, y_encoded)
        self.log("train/loss", loss)
        return loss
    
    def validation_step(self, batch, *args):
        (
            x_row, 
            y_encoded,
            _y_label,
        ) = (
            batch["X"], 
            batch["y"],
            batch["label"],
        )
        
        y_hat = self.model(x_row)
        y_hat_labels = torch.argmax(y_hat, dim=1) + 1

        acc = accuracy(y_hat_labels, _y_label, task='multiclass', num_classes=y_encoded.shape[1])
        self.log("validation/accuracy", acc)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

def train():
    pass_model = CyrusPassClassifier() # create pl.LightningModule object

    trainer = pl.Trainer(
        limit_test_batches=100, 
        max_epochs=200,
        gradient_clip_val=1.0,
        gpus=[0],
        callbacks=[
            ModelCheckpoint(
                monitor="train/loss", 
                filename="cyrus-model-epoch{epoch:02d}-val_loss{train/loss:.2f}", 
                dirpath="./lightning_logs/modelcheckpoint/",
            )
        ]
    ) # create pl.Trainer
    
    trainer.fit(
        model=pass_model,
        datamodule=DataModule(num_workers=2)
    ) # fit the model

    # sample_classifier.save_onnx_model() # save the model

def evaluate():
    pass_model = CyrusPassClassifier.load_from_checkpoint("./lightning_logs/modelcheckpoint/just_pass_loss=1.54.ckpt")
    
    pass_model.eval()
    
    # load data
    basedir_csv = '/home/pedro/robocin/simulation-2d-cpp/py-scripts/action-generation/db/dataset-nader/pass_dataset_test/'
    list_dfs = []
    
    for dirpath, dirnames, filenames in os.walk(basedir_csv):
        for file in filenames:
            if '.csv' not in file:
                continue
            df = pd.read_csv(dirpath + file, index_col=False)
            # print(dirpath + file)
            list_dfs.append(df)
    
    full_df = pd.concat(list_dfs)
    full_df = full_df.drop(columns=["Unnamed: 781"])
    
    yreal = full_df[['out_unum']]
    X_df = full_df.drop(columns=['out_unum'])
    X_values = X_df.values.astype(np.float32) 
    X_tensors = torch.from_numpy(X_values)
    
    y_hat = pass_model(X_tensors)
    y_hat_labels = torch.argmax(y_hat, dim=1) + 1
    
    acc = accuracy(y_hat_labels.reshape(y_hat_labels.size()[0], 1), torch.tensor(yreal.values, dtype=torch.float32), task='multiclass', num_classes=y_hat.shape[1])
    print(f'accuracy is = {acc:.2f}')
    
def main():
    if len(sys.argv) < 2:
        print('Please provide the arguments needed')
        print('mode:{train|evaluate}')
        exit(1)
    
    arg_mode = sys.argv[1]
    
    match arg_mode:
        case "train":
            train()
        case "evaluate":
            evaluate()
        case _ :
            print("Please provide a valid mode")
    

if __name__ == "__main__":
    main()