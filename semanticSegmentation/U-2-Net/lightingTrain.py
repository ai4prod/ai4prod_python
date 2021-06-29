import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from model import U2NET

from torchvision import transforms
from torch.utils.data import random_split, DataLoader

import torch
import torch.nn as nn

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

import os
import glob

class salobj_dataloaderLightning(pl.LightningDataModule):

    def __init__(self, image_list, label_list, batch_size):
        super().__init__()
        self.train_trasform = transforms.Compose([
            RescaleT(320),
            RandomCrop(288),
            ToTensorLab(flag=0)])
        self.image_list = image_list
        self.label_list = label_list
        self.batch_size_train = batch_size

    def setup(self, stage=None):
        self.trainDataset = SalObjDataset(
            img_name_list=self.image_list,
            lbl_name_list=self.label_list,
            transform=self.train_trasform)

    def train_dataloader(self):
        return DataLoader(self.trainDataset, batch_size=self.batch_size_train, shuffle=True, num_workers=1)


class u2Squared(pl.LightningModule):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.model = U2NET(in_ch, out_ch)
        self.bce_loss = nn.BCELoss(size_average=True)

    def multi_bce_loss_fusion(self, d0, d1, d2, d3, d4, d5, d6, labels_v):

        loss0 = self.bce_loss(d0, labels_v)
        loss1 = self.bce_loss(d1, labels_v)
        loss2 = self.bce_loss(d2, labels_v)
        loss3 = self.bce_loss(d3, labels_v)
        loss4 = self.bce_loss(d4, labels_v)
        loss5 = self.bce_loss(d5, labels_v)
        loss6 = self.bce_loss(d6, labels_v)

        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        # print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (loss0.data.item(), loss1.data.item(
        # ), loss2.data.item(), loss3.data.item(), loss4.data.item(), loss5.data.item(), loss6.data.item()))

        return loss0, loss

    def forward(self, x):
        d0, d1, d2, d3, d4, d5, d6 = self.model(x)
        return d0, d1, d2, d3, d4, d5, d6

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, betas=(
            0.9, 0.999), eps=1e-08, weight_decay=0)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch['image'], batch['label']
        labels.type_as(inputs)

        d0, d1, d2, d3, d4, d5, d6 = self.model(inputs)
        loss2, loss = self.multi_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels)
        self.log("train_loss", loss)
        return loss


if __name__ == "__main__":
    
    #data_dir = os.path.join(os.getcwd(), 'train_data' + os.sep)
    data_dir = "/media/ubuntu/dati/Dataset/"
    tra_image_dir = os.path.join('DUTS-TR', 'DUTS-TR-Image' + os.sep)
    tra_label_dir = os.path.join('DUTS-TR', 'DUTS-TR-Mask' + os.sep)

    print(tra_image_dir)
    

    image_ext = '.jpg'
    label_ext = '.png'

    # model saved models
    
    
    epoch_num = 100000
    batch_size = 12
    batch_size_val = 1
    train_num = 0
    val_num = 0

    tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)

    

    tra_lbl_name_list = []
    for img_path in tra_img_name_list:
        img_name = img_path.split(os.sep)[-1]

        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1, len(bbb)):
            imidx = imidx + "." + bbb[i]

        tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)

    print("---")
    print("train images: ", len(tra_img_name_list))
    print("train labels: ", len(tra_lbl_name_list))
    print("---")
    # Dataloader
    input("image")
    salDataloader =salobj_dataloaderLightning(tra_img_name_list,tra_lbl_name_list,batch_size)

    # model initizialiazation
    u2LightingModule = u2Squared(3, 1)

    # this checkpoint monitor train_loss
    checkpoint_callback = ModelCheckpoint(monitor='train_loss',
                                          dirpath="saved_models",
                                          filename="checkpoint",
                                          mode="min",save_top_k=2)
    

    trainer= pl.Trainer(max_epochs=10,gpus=1,progress_bar_refresh_rate=20,callbacks=[checkpoint_callback])
    trainer.fit(model=u2LightingModule,datamodule=salDataloader)
