import time
import torch
import torch.optim as optim
import torch.nn as nn
from torchmetrics.classification import F1Score, Precision, Recall
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score
import pytorch_lightning as pl


class LightningModule(pl.LightningModule):

    def __init__(self, net, n_classes, lr=0.001, optimizer=None, scheduler=None, experiment_name=""):
        super().__init__()

        self.criterion = nn.CrossEntropyLoss()
        self.net = net
        self.learning_rate = lr
        self.n_classes = n_classes
        self.experiment_name = experiment_name

        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        
        self.use_scheduler = False
        if scheduler is not None:
            self.optimizer = {"optimizer": self.optimizer, "lr_scheduler": scheduler}
            self.use_scheduler = True
        
        self.preds = []
        self.targets = []

    def training_step(self, batch, batch_idx):
        _, acc, loss = self._get_preds_acc_loss(batch)

        if self.use_scheduler:
            sch = self.lr_schedulers()
            sch.step()

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):

        preds, acc, loss = self._get_preds_acc_loss(batch)
        self.log("val_loss", loss,  on_epoch=True, prog_bar=True)
        self.log("val_acc", acc,  on_epoch=True, prog_bar=True)

        self.preds.append(preds)
        self.targets.append(batch[1])

        return loss
    
    def _f1_precision_recall_acc(self, average="macro"):

        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets)

        f1 = MulticlassF1Score(num_classes=self.n_classes, average=average).cuda()
        prec = MulticlassPrecision(num_classes=self.n_classes, average=average).cuda()
        recall = MulticlassRecall(num_classes=self.n_classes, average=average).cuda()

        acc = (preds == targets).float().mean()

        return f1(preds, targets), prec(preds, targets), recall(preds, targets), acc

    def on_validation_epoch_end(self):
        f1, prec, rec, acc = self._f1_precision_recall_acc()

        self.log("val_f1", f1, prog_bar=True)
        del self.preds
        del self.targets
        self.preds, self.targets = [], []

    def on_test_epoch_end(self):
        f1, prec, rec, acc = self._f1_precision_recall_acc(average=None)

        t = time.strftime("%Y-%m-%d_%H-%M-%S")
        text = f"{t},{acc},{torch.mean(f1)},{torch.mean(prec)},{torch.mean(rec)},"
        text += ",".join([f"{f1[i]},{prec[i]},{rec[i]}" for i in range(self.n_classes)])
        with open(f"results/{self.experiment_name}_test_resutls.csv", "a") as f:
            print(text, file=f,flush=True)

        del self.preds
        del self.targets
        self.preds, self.targets = [], []

    def test_step(self, batch, batch_idx):
        preds, acc, loss = self._get_preds_acc_loss(batch)
        self.log("test_loss", loss,  on_epoch=True, prog_bar=True)
        self.log("test_acc", acc,  on_epoch=True, prog_bar=True)

        self.preds.append(preds)
        self.targets.append(batch[1])

        return loss
    
    def forward(self, x):
        return self.net(x.float())

    def _get_preds_acc_loss(self, batch):
        x,y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        return preds, acc, loss

    def configure_optimizers(self):
        return self.optimizer