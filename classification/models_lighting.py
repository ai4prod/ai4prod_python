#File for model class of pytorch lightingModule
import pytorch_lightning as pl
from pytorch_lightning import callbacks
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import os
import torchvision
from torchvision import transforms,models
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
import torch.nn.functional as F



class ImagenetTransferLearning(pl.LightningModule):
    """
    Class to Finetune Resnet50.
    Resnet50 exploited is the torchvision one
    """

    def __init__(self, num_classes):
        super().__init__()
        # save init parameters in a dictionary. Use by load_from_checkpoint
        self.save_hyperparameters()
        self.num_classes = num_classes
        # init pretrained
        backbone = torchvision.models.resnet18(pretrained=True)
        num_param = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = torch.nn.Sequential(*layers)
        self.finetunelayer = torch.nn.Linear(num_param, self.num_classes)
        
         
        
        
    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.finetunelayer(representations)
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        train_acc = accuracy(logits, y)
        self.log('train_loss', loss)
        self.log('train_accuracy', train_acc, prog_bar=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            preds = self(x)
            val_loss = F.cross_entropy(preds, y)
            val_acc = accuracy(preds, y)
            self.log('val_loss', val_loss)
            self.log('val_accuracy', val_acc, prog_bar=True, on_epoch=True)
        return {'loss': val_loss, 'accuracy': val_acc}
    
    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.tensor([x['loss'] for x in outputs]).mean()
        avg_acc = sum([x['accuracy'] for x in outputs]).mean()
        self.log('val_loss', avg_val_loss)
        self.log('val_accuracy', avg_acc)
    
  




#class to train Imagenet Dataset

class ImagenetInference(pl.LightningModule):

  

    def __init__(self,model_path=None):
        
        super().__init__()
        self. transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        if model_path:
            self.model= torch.load(model_path)
        else:
            self.model= models.resnet50(pretrained=True)

    def forward(self,x):

        self.model.eval()
        out= self.model(x)
        return out