from torch import optim,cuda
from torchvision import models,transforms
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb
from pytorch_lightning.loggers import WandbLogger
from models.resnet9 import ResNet9
from classifier import classifier
from pl_bolts.datamodules import CIFAR10DataModule
import torch

net=ResNet9(3,10)
teacher_net=ResNet9(3,10)
wandb.login(key='bccba7e310a012fecaf8352d16b4c8829e513214')
model_name=f'resnet9'
wandb_logger = WandbLogger(project="mnist",name=model_name, log_model="all")
save_best_cb= ModelCheckpoint(
    monitor='val_acc',
    filename='{epoch:03d}-{val_acc:.4f}',
    save_last=True,
    mode='max',
)
trainer=pl.Trainer(
    accelerator='auto',
    gpus=1 if cuda.is_available() else 0,
    max_epochs=100,
    logger=wandb_logger,
    callbacks=[save_best_cb],
)
teacher=classifier(teacher_net,teacher=None)
# teacher.load_from_checkpoint('./dl_cv/best_models/epoch=020-val_acc=0.7875.ckpt',strict=False)
path='./best_models/epoch=096-val_acc=0.8275.ckpt'
teacher.load_state_dict(torch.load(path)['state_dict'])
student=classifier(net,teacher=teacher)
cifar10=CIFAR10DataModule()
trainer.fit(student,datamodule=cifar10)