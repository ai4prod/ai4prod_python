import torch
import onnx

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from model import U2NET # full size version 173.6 MB

HEIGHT=320
WIDTH=320

dymmy_input= torch.rand((1,3,WIDTH,HEIGHT))

net = U2NET(3,1)
model_dir="u2net.pth"
net.load_state_dict(torch.load(model_dir, map_location='cpu'))


torch.onnx.export(net,               # model being run
                  dymmy_input,                         # model input (or a tuple for multiple inputs)
                  "u2net.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  verbose=True)

print("CHEK MODEL")
onnx_model = onnx.load("u2net.onnx")
onnx.checker.check_model(onnx_model)