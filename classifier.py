from torch import optim
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy

class classifier(pl.LightningModule):
    def __init__(self,net,teacher=None):
        super().__init__()
        self.net=net
        self.loss_fn=nn.CrossEntropyLoss()
        self.train_acc=Accuracy(num_classes=10,multiclass=True)
        self.val_acc=Accuracy(num_classes=10,multiclass=True)
        self.teacher=teacher

    def forward(self,x):
        out=self.net(x)
        return out
    def training_step(self, batch, batch_idx):
        x, y = batch
        print(y)
        out = self.net(x)
        if self.teacher!=None:
            y=self.teacher(x)
        print(y)
        loss = self.loss_fn(out, y)
        acc=self.train_acc(out,y)
        self.log('train_loss',loss,prog_bar=True,on_epoch=True,on_step=False)
        self.log('train_acc',acc,prog_bar=True,on_epoch=True,on_step=False)
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.net(x)
        loss = self.loss_fn(out, y)
        acc=self.val_acc(out,y)
        self.log('val_loss',loss,prog_bar=True,on_epoch=True,on_step=False)
        self.log('val_acc',acc,prog_bar=True,on_epoch=True,on_step=False)
        return loss
    def configure_optimizers(self):
        optimizer=optim.SGD(self.net.parameters(),lr=0.05,momentum=0.9)
        return optimizer
