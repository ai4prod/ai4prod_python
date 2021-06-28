import pytorch_lightning as pl
from model import U2NET

import torch
import torch.nn as nn

class u2Squared(pl.LightningModule):
    
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.model= U2NET(in_ch,out_ch)
        self.bce_loss = nn.BCELoss(size_average=True)
    
    def muti_bce_loss_fusion(self,d0, d1, d2, d3, d4, d5, d6, labels_v):

        loss0 = self.bce_loss(d0,labels_v)
        loss1 = self.bce_loss(d1,labels_v)
        loss2 = self.bce_loss(d2,labels_v)
        loss3 = self.bce_loss(d3,labels_v)
        loss4 = self.bce_loss(d4,labels_v)
        loss5 = self.bce_loss(d5,labels_v)
        loss6 = self.bce_loss(d6,labels_v)

        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

	return loss0, loss
    
    def forward(self,x):
        d0, d1, d2, d3, d4, d5, d6 = self.model(x)
        return d0, d1, d2, d3, d4, d5, d6

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)


    def training_step(self,batch,batch_idx):
        inputs, labels = batch['image'], batch['label']
        d0, d1, d2, d3, d4, d5, d6 = self.model(inputs)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels)
        self.log("train_loss",loss)
        return loss
        


if __name__ =="__main__":


    #Dataloader

    

    # model initizialiazation
    u2LightingModule= u2Squared(1,3)

