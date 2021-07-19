#from trainConfMatrix import Net
import torch
import torchvision
from torchvision import transforms


from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import pytorch_lightning as pl 
import torch
from datamodule_lighting import *
from models_lighting import *

#Create confusion Matrix 

# net= torchvision.models.resnet50(pretrained=True)

# net.cuda()
# net.eval()

# transform = transforms.Compose([
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])])


# testset = torchvision.datasets.ImageFolder('./test_cifar10',
#     transform=transform)

# testloader = torch.utils.data.DataLoader(testset, batch_size=8,
#                                         shuffle=False, num_workers=2)


# y_pred = []
# y_true = []

# for inputs, labels in testloader:
#         output = net(inputs.cuda()) # Feed Network

#         output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
#         print(output)
#         input("te")
#         y_pred.extend(output) # Save Prediction
        
#         labels = labels.data.cpu().numpy()
        
        
#         y_true.extend(labels) # Save Truth


# print(len(y_pred))
# print(len(y_true))
# # constant for classes
# classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
#         'dog', 'frog', 'horse', 'ship', 'truck')

# # Build confusion matrix
# cf_matrix = confusion_matrix(y_true, y_pred)
# df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix)*10 , index = [i for i in classes],
#                      columns = [i for i in classes])

# df_cm.to_csv("confusionMatrixCifar10.csv",index=False,header=False)
# plt.figure(figsize = (12,7))
# sn.heatmap(df_cm, annot=True)
# plt.savefig('outputCifar10.png')



#lighting confusion Matrix
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

dm = ImageFolderTransferLearning(
        "/media/aistudios/44c62318-a7de-4fb6-a3e2-01aba49489c5/Dataset/cifar-10-python/cifar-10-batches-py", batch_size=12, train_transform=train_transform, test_transform=test_transform)

dm.setup()
valDataloader= dm.val_dataloader()



model=ImagenetTransferLearning.load_from_checkpoint("trained_models/finetuned-resnet-cifar.ckpt")

model.eval()
model.cuda()


y_pred = []
y_true = []

for data in valDataloader:
    tensor,labels=data
    tensor = tensor.cuda()
    #print(labels)
    output = model.forward(tensor)
    #model forward is without softmax
    output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
    y_pred.extend(output)

    labels = labels.data.cpu().numpy()
    y_true.extend(labels)
    #print(output)


classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck')

# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix)*10 , index = [i for i in classes],
                     columns = [i for i in classes])

df_cm.to_csv("confusionMatrixCifar10Scratch.csv",index=False,header=False)
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.savefig('outputCifar10Real.png')