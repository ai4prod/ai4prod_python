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
import onnx

class ImagenetTransferLearning(pl.LightningModule):
    """
    Class to Finetune Resnet50.
    Resnet50 exploited is the torchvision one
    """

    def __init__(self, num_classes):
        super().__init__()
        self. transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.num_classes = num_classes
        # init pretrained
        backbone = torchvision.models.resnet50(pretrained=True)
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
    
    def train_dataloader(self):
        trainloader = torch.utils.data.DataLoader(CIFAR10(root='./data', train=True,
                                        download=True, transform=self.transform), 
                                        batch_size=8,
                                        shuffle=True, num_workers=4)
        return trainloader
    
    def val_dataloader(self):
        testloader = torch.utils.data.DataLoader(CIFAR10(root='./data', train=False,
                                       download=True, transform=self.transform), 
                                       batch_size=8,
                                       shuffle=False, num_workers=4)
        return testloader
        
        
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
    
    # Training
    model = ImagenetTransferLearning(num_classes=10)
    trainer = pl.Trainer(max_epochs=1, gpus=1, progress_bar_refresh_rate=20, callbacks=[checkpoint_callback])
    trainer.fit(model)
    torch.save(model.state_dict(), 'finetune.pth')

    # load model
    model2onnx = ImagenetTransferLearning(num_classes=10)
    model2onnx.load_state_dict(torch.load('finetune.pth', map_location = 'cpu'))
    model2onnx.eval()
    dummy_input_real = torch.randn(1, args.channels, args.imsize, args.imsize)
    torch.onnx.export(model2onnx,
                        dummy_input_real,
                        'resnet50finetuned.onnx',
                        opset_version=11,
                        do_constant_folding=True,
                        export_params=True
                        )
    
    onnx_model = onnx.load("resnet50finetuned.onnx")
    onnx.checker.check_model(onnx_model)

    print("----Conversion Completed----")
    
