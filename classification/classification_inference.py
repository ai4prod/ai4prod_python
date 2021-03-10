import pytorch_lightning as pl 
import torch
from datamodule_lighting import *
from models_lighting import *

testDataLoader= ImageFolderTransferLearningTest("data")

testDataLoader.setup()

DataLoader=testDataLoader.test_dataloader()

model=ImagenetTransferLearning.load_from_checkpoint("trained_models/finetuned-resnet.ckpt")

model.eval()
model.cuda()

for data in DataLoader:
    tensor,label=data
    tensor = tensor.cuda()
    
    out= model(tensor)

    



