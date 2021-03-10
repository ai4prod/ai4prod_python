# custom Datamodule for training


import pytorch_lightning as pl
from torchvision import transforms,datasets

from torch.utils.data import random_split, DataLoader 

#Data Module for transfer learning from Imagenet

class ImageFolderTransferLearning(pl.LightningDataModule):
    def __init__(self,datasetPath):
        super().__init__()
        self.train_transform= transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.test_transform= transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.datasetPath=datasetPath

    def setup(self,stage=None):
        
        self.trainDataset= datasets.ImageFolder(self.datasetPath+ "/train",transform=self.train_transform)
        self.valDataset= datasets.ImageFolder(self.datasetPath+ "/val",transform=self.test_transform)
    
    def train_dataloader(self):

        return DataLoader(self.trainDataset,batch_size=8,shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.valDataset,batch_size=8,shuffle=True)

class ImageFolderTransferLearningTest(pl.LightningDataModule):
    def __init__(self,datasetPath):
        super().__init__()
        self.test_transforms= transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.datasetPath=datasetPath
    
    def setup(self,stage=None):
        
        self.testDataset= datasets.ImageFolder(self.datasetPath+ "/test",transform=self.test_transforms)
    
    def test_dataloader(self):

        return DataLoader(self.testDataset,batch_size=1,shuffle=True)
       

        
