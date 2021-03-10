import pytorch_lightning as pl 
import torch
from datamodule_lighting import *
from models_lighting import *


normalize_data=[0.485,0.456,0.406,0.229,0.224,0.225]

testDataLoader= ImageFolderTransferLearningTest("data",img_w=224,img_h=224,normValue=normalize_data,batch_size=1)

testDataLoader.setup()

DataLoader=testDataLoader.test_dataloader()

model=ImagenetTransferLearning.load_from_checkpoint("trained_models/finetuned-resnet.ckpt")

model.eval()
model.cuda()

for data in DataLoader:
    tensor,label=data
    tensor = tensor.cuda()
    print(label)
    #model forward is without softmax
    out= model(tensor)
    print(out)
    
    



