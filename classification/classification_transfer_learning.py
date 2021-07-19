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
#remove warning message
# import logging
# logging.getLogger("lightning").setLevel(logging.ERROR)

if __name__ == "__main__":

    parser = ArgumentParser()
    # automaticaly parse all the things you want
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--batchsize', default=4)
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
        dirpath=model_dir,
        filename='finetuned-resnet-cifar',
        save_top_k=1,
        mode='min',
    )

    # ImageNet Transform
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    #
    # train_transform = transforms.Compose([
    #     transforms.RandomResizedCrop(224),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     normalize
    #     ])

    # test_transform = transforms.Compose([
    #     transforms.Resize((256,256)),
    #     transforms.CenterCrop([224,224]),
    #     transforms.ToTensor(),
    #     normalize
    #     ])

    # CIFAR Transform
    normalize = transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # Training
    # Img_w Img_h batch_size NEED TO CHANGE WITH CUSTOM VALUE
    dm = ImageFolderTransferLearning(
        "/media/aistudios/44c62318-a7de-4fb6-a3e2-01aba49489c5/Dataset/cifar-10-python/cifar-10-batches-py", batch_size=128, train_transform=train_transform, test_transform=test_transform)

    # you can change with any model you want
    model= torchvision.models.resnet18()
    
    # NUM_CLASSES NEED TO CHANGE WITH CUSTOM VALUE
    model = ImagenetTransferLearning(num_classes=10,pytorch_model=model,from_scratch=True)

    trainer = pl.Trainer(max_epochs=50, gpus=1, progress_bar_refresh_rate=20,precision=16, callbacks=[
                         checkpoint_callback])
    trainer.fit(model=model, datamodule=dm)
