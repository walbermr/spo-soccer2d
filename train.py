import pytorch_lightning as pl

from feature_sets import *

from datamodule import DataModule

from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from model import SPO

from params import hparams, load_params


def main():
    
    logger = TensorBoardLogger(save_dir=".", name="training_logs")

    checkpoint_callback = ModelCheckpoint(
        filename="{epoch}", save_top_k=2, monitor="validation/evaluator/top1_correct", mode="max"
    )

    trainer = pl.Trainer(
        logger=logger,
        limit_train_batches=1.0, 
        limit_test_batches=1.0, 
        max_epochs=100,
        gradient_clip_val=1.0,
        detect_anomaly=False,
        callbacks=[checkpoint_callback],
        limit_val_batches=1.0,
        accelerator="gpu",
        devices=[0],
    ) # create pl.Trainer

    datamodule = DataModule(
        num_workers=1, 
        normalize_dset=hparams["dataset"]["normalize_dset"], 
        normalize_evaluation=hparams["dataset"]["normalize_evaluation"],
        batch_size=100, 
        feature_set=hparams["dataset"]["fset"],
    )
    
    pass_model = SPO(
        hparams=hparams,
    ) # create pl.LightningModule object
    
    trainer.fit(
        model=pass_model,
        datamodule=datamodule,
    ) # fit the model

    trainer.test(dataloaders=datamodule)


if __name__ == "__main__":
    load_params()
    main()
