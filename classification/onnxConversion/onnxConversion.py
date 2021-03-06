import sys,os
#need to include file in parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import onnx

from models_lighting import *
from torchvision import models

#ATTENTION Change this value based on your training
#Imagenet
# image_channel_in= 3 #RGB
# image_size_w=224
# image_size_h=224

#Cifar10
image_channel_in= 3 #RGB
image_size_w=32
image_size_h=32


#ATTENTION substitute with the pytorch-lighting class that you have trained. Have a look inside models_lighting.py
model=ImagenetTransferLearning.load_from_checkpoint("../trained_models/finetuned-resnet-cifar.ckpt")

dummy_input_real = torch.randn(1, image_channel_in, image_size_w, image_size_h)

model.to_onnx("models/finetuned-resnet-cifar.onnx",dummy_input_real,opset_version=11,do_constant_folding=True,export_params=True,output_names=["output1"])

