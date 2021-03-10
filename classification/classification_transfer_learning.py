import pytorch_lightning as pl
from pytorch_lightning import callbacks
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import os
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
import torch.nn.functional as F
from argparse import ArgumentParser
from models_lighting import *
from datamodule_lighting import *


        
if __name__ == "__main__":
    
    parser = ArgumentParser()
    # automaticaly parse all the things you want
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--batchsize', default = 4)
    parser.add_argument('--imsize', default=224, 
                            help='Image size for ONNX conversion and Inference')
    parser.add_argument('--channels', default=3, 
                            help='Image channels for ONNX conversion and Inference')                       
    args = parser.parse_args()
    
    model_dir = os.path.join(".", "trained_models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath= model_dir,
    filename='finetuned-resnet',
    save_top_k=1,
    mode='min',
    )
    
    #NEED TO CHANGE WITH YOUR VALUE
    normalize_data=[0.485,0.456,0.406,0.229,0.224,0.225]

    # Training
    #Img_w Img_h batch_size NEED TO CHANGE WITH CUSTOM VALUE
    dm= ImageFolderTransferLearning("data",img_w=224,img_h=224,normValue=normalize_data,batch_size=8)
    
    #NUM_CLASSES NEED TO CHANGE WITH CUSTOM VALUE
    model = ImagenetTransferLearning(num_classes=2)
    
    trainer = pl.Trainer(max_epochs=10, gpus=1, progress_bar_refresh_rate=20, callbacks=[checkpoint_callback])
    trainer.fit(model=model,datamodule=dm)
    
