# custom Datamodule for training


import pytorch_lightning as pl
from torchvision import transforms,datasets

from torch.utils.data import random_split, DataLoader 

#Data Module for transfer learning from Imagenet
#Images need to be in RGB order for this normalize value
class ImageFolderTransferLearning(pl.LightningDataModule):
    def __init__(self,datasetPath,batch_size,train_transform,test_transform):
        super().__init__()
        self.train_transform= train_transform
        self.test_transform= test_transform

        self.datasetPath=datasetPath
        self.batch_size=batch_size

    def setup(self,stage=None):
        
        self.trainDataset= datasets.ImageFolder(self.datasetPath+ "/train",transform=self.train_transform)
        self.valDataset= datasets.ImageFolder(self.datasetPath+ "/val",transform=self.test_transform)
    
    def train_dataloader(self):

        return DataLoader(self.trainDataset,batch_size=self.batch_size,shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.valDataset,batch_size=self.batch_size,shuffle=False)

# DataModule for testing ImageFolderTransferLearning
# here we load only test set
class ImageFolderTransferLearningTest(pl.LightningDataModule):
    def __init__(self,datasetPath, img_w,img_h,normValue,batch_size):
        super().__init__()
        self.test_transforms= transforms.Compose([
        transforms.Resize([img_w,img_h]),
        transforms.ToTensor(),
        transforms.Normalize((normValue[0], normValue[1], normValue[2]), (normValue[3], normValue[4], normValue[5]))
        ])
        self.datasetPath=datasetPath
        self.batch_size=batch_size
    
    def setup(self,stage=None):
        
        self.testDataset= datasets.ImageFolder(self.datasetPath+ "/test",transform=self.test_transforms)
    
    def test_dataloader(self):

        return DataLoader(self.testDataset,batch_size=self.batch_size,shuffle=False)
       

        
